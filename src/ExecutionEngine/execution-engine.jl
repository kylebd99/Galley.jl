# This file performs the actual execution of physical query plan.
function initialize_access(tensor_id::TensorId, tensor, index_ids, protocols::Vector{AccessProtocol}; read=true)
    mode = read ? Reader() : Updater()
    mode = literal_instance(mode)
    index_expressions = []
    for i in range(1, length(index_ids))
        index = index_instance(Symbol(index_ids[i]))
        if read == true
            if protocols[i] == t_walk
                index = call_instance(literal_instance(Finch.walk), index)
            elseif protocols[i] == t_gallop
                index = call_instance(literal_instance(Finch.gallop), index)
            elseif protocols[i] == t_lead
                index = call_instance(literal_instance(Finch.lead), index)
            elseif protocols[i] == t_follow
                index = call_instance(literal_instance(Finch.follow), index)
            end
        end
        push!(index_expressions, index)
    end
    tensor_var = variable_instance(Symbol(tensor_id))
    tensor_tag = tag_instance(tensor_var, tensor)
    tensor_access = access_instance(tensor_tag, mode, index_expressions...)
    return tensor_access
end


function execute_tensor_kernel(kernel::TensorKernel; lvl = 1, verbose=0)
    for tensor_id in keys(kernel.input_tensors)
        if kernel.input_tensors[tensor_id] isa TensorKernel
            kernel.input_tensors[tensor_id] = execute_tensor_kernel(kernel.input_tensors[tensor_id],
                                                                     lvl=lvl+1,
                                                                     verbose=verbose)
        end
    end

    verbose >= 1 && println("------- Kernel at Depth: $(lvl) -------")
    nodes_to_visit = Queue{Tuple{TensorExpression, Int64}}()
    node_dict = Dict()
    node_id_counter = 0
    enqueue!(nodes_to_visit, (kernel.kernel_root, node_id_counter))
    while length(nodes_to_visit) > 0
        cur_node, cur_node_id = dequeue!(nodes_to_visit)
        child_node_ids = []
        if cur_node isa OperatorExpr
            for child_node in cur_node.inputs
                node_id_counter += 1
                enqueue!(nodes_to_visit, (child_node, node_id_counter))
                push!(child_node_ids, node_id_counter)
            end
        elseif cur_node isa AggregateExpr || cur_node isa ReorderExpr
            node_id_counter += 1
            enqueue!(nodes_to_visit, (cur_node.input, node_id_counter))
            push!(child_node_ids, node_id_counter)
        end
        node_dict[cur_node_id] = (cur_node, child_node_ids)
    end

    agg_op = nothing
    kernel_prgm = nothing
    input_index_orders = []
    for node_id in reverse(range(0, length(keys(node_dict))-1))
        node, child_node_ids = node_dict[node_id]
        if node isa InputExpr
            tensor_id = node.tensor_id
            if kernel.input_tensors[tensor_id] isa Number
                node_dict[node_id] = literal_instance(kernel.input_tensors[tensor_id])
            else
                if verbose >= 3
                    println("Stats Type: ", typeof(node.stats))
                    println("Indices:  ", get_index_set(node.stats))
                    println("Expected Output Tensor Size: ", estimate_nnz(node.stats))
                    println("Output Tensor Size: ", countstored(kernel.input_tensors[tensor_id]))
                end
                node_dict[node_id] = initialize_access(tensor_id,
                                                        kernel.input_tensors[tensor_id],
                                                        node.input_indices,
                                                        node.input_protocols)
                push!(input_index_orders, node.input_indices)
            end
        elseif node isa OperatorExpr
            child_prgms = [node_dict[x] for x in child_node_ids]
            op = node.op
            node_dict[node_id] = call_instance(literal_instance(op), child_prgms...)
        end
        if node_id == 0
            if node isa AggregateExpr
                kernel_prgm = node_dict[child_node_ids[1]]
                agg_op = node.op
            elseif node isa ReorderExpr
                kernel_prgm = node_dict[child_node_ids[1]]
            else
                kernel_prgm = node_dict[node_id]
            end
        elseif node isa AggregateExpr
            throw(ArgumentError("Cannot have an aggregate in the middle of a tensor kernel. They must always occur as the outermost operator."))
        elseif node isa ReorderExpr
            throw(ArgumentError("Cannot have an reorder in the middle of a tensor kernel. They must always occur as the outermost operator."))
        end
    end

    output_dimensions = kernel.output_dims
    output_default = kernel.output_default
    output_tensor = initialize_tensor(kernel.output_formats,
                                        output_dimensions,
                                        output_default)
    output_access = initialize_access("output_tensor",
                                        output_tensor,
                                        kernel.output_indices,
                                        [t_default for _ in kernel.output_indices];
                                        read=false)

    if agg_op === nothing
        agg_op = initwrite(output_default)
    end

    full_prgm = assign_instance(output_access, literal_instance(agg_op), kernel_prgm)

    loop_order = [index_instance(Symbol(i)) for i in kernel.loop_order]
    for index in reverse(loop_order)
        full_prgm = loop_instance(index, Dimensionless(), full_prgm)
    end
    full_prgm = block_instance(declare_instance(variable_instance(:output_tensor),
                                                 literal_instance(output_default)),
                                full_prgm)
    printKernel(kernel, verbose)
    start_time = time()
    Finch.execute(full_prgm, (mode=Finch.FastFinch(),))
    verbose >= 2 && println("Kernel Execution Took: ", time() - start_time)
    if output_tensor isa Finch.Scalar
        return output_tensor[]
    else
        return output_tensor
    end
end
