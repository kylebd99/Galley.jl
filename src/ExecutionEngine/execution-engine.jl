# To reduce compilation overhead, we try and roughly cannonicalize the inputs
# to operator_expr by sorting them.
function sort_mapjoin_args(args)
    immediate_args = [arg for arg in args if arg.kind == Input || arg.kind == Alias]
    remainder = [arg for arg in args if !(arg.kind == Input || arg.kind == Alias)]
    perm = sortperm([(length(get_index_order(arg.stats)), get_index_formats(arg.stats)..., get_index_protocols(arg.stats)...,get_index_order(arg.stats)) for arg in immediate_args])
    return [immediate_args[perm]..., remainder...]
end

function translate_rhs(alias_dict, tensor_counter, index_sym_dict, rhs::PlanNode)
    if rhs.kind == Alias
        tns = alias_dict[rhs.name]
        idxs = get_index_order(rhs.stats)
        @assert all([get_dim_size(rhs.stats, idxs[i]) == size(tns)[i] for i in eachindex(idxs)]) "$(size(tns)) $(idxs) $([(X, Int64(x)) for (X,x) in rhs.stats.def.dim_sizes])"
        protocols = [get_index_protocol(rhs.stats, idx) for idx in idxs]
        t_name = get_tensor_symbol(tensor_counter[1])
        tensor_counter[1] += 1
        return initialize_access(t_name, tns, idxs, protocols, index_sym_dict, read=true)

    elseif rhs.kind === Input
        idxs = get_index_order(rhs.stats)
        protocols = [get_index_protocol(rhs.stats, idx) for idx in idxs]
        t_name = get_tensor_symbol(tensor_counter[1])
        tensor_counter[1] += 1
        return initialize_access(t_name, rhs.tns.val, idxs, protocols, index_sym_dict, read=true)
    elseif rhs.kind == Value
        if rhs.val isa Number
            return literal_instance(rhs.val)
        end
    elseif rhs.kind === MapJoin
        if iscommutative(rhs.op.val)
            rhs.args = sort_mapjoin_args(rhs.args)
        end
        if is_binary(rhs.op.val)
            instance = translate_rhs(alias_dict, tensor_counter, index_sym_dict, rhs.args[1])
            for arg in rhs.args[2:end]
                instance = call_instance(literal_instance(rhs.op.val), translate_rhs(alias_dict, tensor_counter, index_sym_dict, arg), instance)
            end
            return instance
        else
            return call_instance(literal_instance(rhs.op.val),
                                    [translate_rhs(alias_dict, tensor_counter, index_sym_dict, arg) for arg in rhs.args]...)
        end
    else
        throw(ErrorException("RHS expression cannot contain anything except Alias, Input, and MapJoin: $rhs"))
    end
end

# To be executed, a query must be in the following format:
# Query(name, Materialize(formats..., index_order..., Aggregate(op, idxs..., map_expr)))
# TODO: use loop_order to label indexes
function execute_query(alias_dict, q::PlanNode, verbose)
    tensor_counter = [0]
    index_sym_dict = Dict{IndexExpr, IndexExpr}()
    name = q.name.name
    mat_expr = q.expr
    loop_order = [idx.name for idx in q.loop_order]
    output_formats = [f.val for f in mat_expr.formats]
    output_idx_order = [idx.name for idx in mat_expr.idx_order]
    agg_expr = mat_expr.expr
    output_default = get_default_value(agg_expr.stats)
    output_dimensions = [get_dim_size(agg_expr.stats, idx) for idx in output_idx_order]
    agg_op = agg_expr.op.val
    agg_idxs = [idx.name for idx in agg_expr.idxs]
    rhs_expr = agg_expr.arg
    rhs_instance = translate_rhs(alias_dict, tensor_counter, index_sym_dict, rhs_expr)

    output_tensor = initialize_tensor(output_formats,
                                        output_dimensions,
                                        output_default)
    output_access = initialize_access(:output_tensor,
                                        output_tensor,
                                        output_idx_order,
                                        [t_default for _ in output_idx_order],
                                        index_sym_dict;
                                        read=false)

    dec_instance = declare_instance(variable_instance(:output_tensor),
                                                 literal_instance(output_default))

    prgm_instance = assign_instance(output_access, literal_instance(agg_op), rhs_instance)
    loop_order = [index_instance(index_sym_dict[i]) for i in loop_order]
    for index in reverse(loop_order)
        prgm_instance = loop_instance(index, Dimensionless(), prgm_instance)
    end
    prgm_instance = block_instance(dec_instance, prgm_instance)

    start_time = time()
    verbose >= 4 && display(prgm_instance)
    verbose >= 5 &&  println(Finch.execute_code(:ex, typeof(prgm_instance), mode=:fast)
                                                                |> Finch.pretty
                                                                |>  Finch.unresolve
                                                                |>  Finch.dataflow
                                                                |>  Finch.unquote_literals)
    verbose >= 2 && println("Expected Output Size: $(estimate_nnz(agg_expr.stats))")
    Finch.execute(prgm_instance, mode=:fast)
    verbose >= 2 && println("Kernel Execution Took: ", time() - start_time)
    if output_tensor isa Finch.Scalar
        verbose >= 2 && println("Output Size: 1")
        alias_dict[name] = output_tensor[]
    else
        # There are cases where default entries will be stored explicitly, so we avoid that
        # by re-copying the data. We also check to see if the format should be changed
        # based on the true cardinality.
        touch_up_start = time()
        non_default = count_non_default(output_tensor)
        stored = count_stored(output_tensor)
        estimated_size = estimate_nnz(mat_expr.stats)
        verbose >= 2 && println("Stored Entries: ", stored)
        verbose >= 2 && println("Estimated Entries: ", estimated_size)
        verbose >= 2 && println("Non Default Entries: ", non_default)
        if (stored > (2 * non_default)) || (non_default > 5 * estimated_size) ||(non_default < estimated_size / 5)
            fix_cardinality!(mat_expr.stats, non_default)
            best_formats = select_output_format(mat_expr.stats, reverse(get_index_order(mat_expr.stats)), get_index_order(mat_expr.stats))
            verbose >= 2 &&  println("Touching Up (best_formats: $best_formats, old_formats: $output_formats)")
            if output_formats != best_formats
                output_tensor = initialize_tensor(best_formats,
                                            output_dimensions,
                                            output_default,
                                            copy_data = output_tensor)
                q.expr.formats = [Value(f) for f in best_formats]
                get_def(q.expr.stats).level_formats = best_formats
            end
        end
        verbose >= 2 && println("Touch Up Time: ", time()-touch_up_start)
        alias_dict[name] = output_tensor
    end
end
