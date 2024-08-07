
# This function takes in a dict of (tensor_id => tensor_stats) and outputs a join order.
# Currently, it uses a simple prefix-join heuristic, but in the future it will be cost-based.
function get_join_loop_order_simple(input_stats)
    num_occurrences = counter(IndexExpr)
    for stats in values(input_stats)
        for v in get_index_set(stats)
            inc!(num_occurrences, v)
        end
    end
    vars_and_counts = sort([(num_occurrences[v], v) for v in keys(num_occurrences)], by=(x)->x[1], rev=true)
    vars = [x[2] for x in vars_and_counts]
    return vars
end

cost_of_reformat(stat::TensorStats) = estimate_nnz(stat) * RandomWriteCost

function get_reformat_set(input_stats::Vector{TensorStats}, prefix::Vector{IndexExpr})
    ref_set = Set()
    for i in eachindex(input_stats)
        index_order = get_index_order(input_stats[i])
        # Tensors are stored in column major, so we reverse the index order here
        current_loop = 0
        needs_reformat = false
        for idx in reverse(index_order)
            idx_loop = Inf
            if idx ∈ prefix
                idx_loop = only(indexin([idx], prefix))
            end
            if idx_loop < current_loop
                needs_reformat = true
            end
            current_loop = idx_loop
        end
        needs_reformat && push!(ref_set, i)
    end
    return ref_set
end

# If output_vars is a vector, we assume that we will have to reindex it if it's just a set
# prefix.
function get_output_compat(output_vars::Vector{IndexExpr}, prefix::Vector{IndexExpr})
    return if fully_compat_with_loop_prefix(output_vars, prefix)
        FULL_PREFIX
    elseif set_compat_with_loop_prefix(Set(output_vars), prefix)
        SET_PREFIX
    else
        NO_PREFIX
    end
end

# If output_vars is a set, we don't assume that we'll have to reindex it, so we treat any
# prefix as a FULL_PREFIX.
function get_output_compat(output_vars::Set{IndexExpr}, prefix::Vector{IndexExpr})
    return if set_compat_with_loop_prefix(output_vars, prefix)
        FULL_PREFIX
    else
        NO_PREFIX
    end
end


@enum OUTPUT_COMPAT FULL_PREFIX=0 SET_PREFIX=1 NO_PREFIX=2
PLAN_CLASS = Tuple{Set{IndexExpr}, OUTPUT_COMPAT, Set{Int}}
PLAN = Tuple{Vector{IndexExpr}, Float64}

function cost_of_plan_class(pc::PLAN_CLASS, reformat_costs, output_size)
    output_compat = pc[2]
    rf_set = pc[3]
    pc_cost = 0
    if output_compat == FULL_PREFIX
        pc_cost += output_size * SeqWriteCost
    elseif output_compat == SET_PREFIX
        pc_cost += output_size * SeqWriteCost
    else
        # Theoretically, this should be num_flops * RandomWriteCost, but we've seen that
        # lead to aggressive transposing of inputs.
        pc_cost += output_size * RandomWriteCost
    end
    for i in rf_set
        pc_cost += reformat_costs[i]
    end
    return pc_cost
end

# We use a version of Selinger's algorithm to determine the join loop ordering.
# For each subset of variables, we calculate their optimal ordering via dynamic programming.
# This is singly exponential in the number of loop variables, and it uses the statistics to
# determine the costs.
# `input_stats` is for the set of `Input` and `Alias` expressions,
# `join_stats` reflects the compute tensor (i.e. the set of necessary FLOPs)
# `output_stats` reflects the materialization tensor (i.e. the size of the intermediate produced)
# `output_order` may or may not be present and reflects whether we have a set output order
#  we care about. This will generally be present in the final query to be evaluated.
# TODO: Our cost model is off here. We compute the join stats once then just evaluate
# prefixes relative to that. However, this could be much smaller than the iterations in a
# loop prefix. For example, MapJoin(*, A[i,j], B[i,j], C[j]) might have a join stat with
# size 1 if |C| = 1. In this case, we view both orders to be equal even though j,i is
# potentially O(n) better.
function get_join_loop_order_bounded(disjunct_and_conjunct_stats,
                                    output_stats::TensorStats,
                                    output_order::Union{Nothing, Vector{IndexExpr}},
                                    cost_bound,
                                    top_k)
    disjunct_stats = disjunct_and_conjunct_stats.disjuncts
    conjunct_stats = disjunct_and_conjunct_stats.conjuncts
    all_stats = TensorStats[disjunct_stats..., conjunct_stats...]
    all_vars = union([get_index_set(stat) for stat in all_stats]...)
    output_size = estimate_nnz(output_stats)
    output_vars = get_index_set(output_stats)

    output_vars = get_index_set(output_stats)
    if !isnothing(output_order)
        output_vars = relative_sort(output_vars, output_order)
    end

    # At all times, we keep track of the best plans for each level of output compatability.
    # This will let us consider the cost of random writes and transposes at the end.
    reformat_costs = Dict(i => cost_of_reformat(all_stats[i]) for i in eachindex(all_stats))
    PLAN_CLASS = Tuple{Set{IndexExpr}, OUTPUT_COMPAT, Set{Int}}
    PLAN = Tuple{Vector{IndexExpr}, Float64}
    optimal_plans = Dict{PLAN_CLASS, PLAN}()
    for var in all_vars
        prefix = [var]
        v_set = Set(prefix)
        rf_set = get_reformat_set(all_stats, prefix)
        output_compat = get_output_compat(output_vars, prefix)
        class = (v_set, output_compat, rf_set)
        cost = get_prefix_cost(v_set, conjunct_stats, disjunct_stats)
        optimal_plans[class] = (prefix, cost)
    end

    for _ in 2:length(all_vars)
        new_plans =  Dict{PLAN_CLASS, PLAN}()
        for (plan_class, plan) in optimal_plans
            prefix_set = plan_class[1]
            prefix = plan[1]
            cost = plan[2]
            # We only consider extensions that don't result in cross products
            potential_vars = Set{IndexExpr}()
            for stat in all_stats
                index_set = get_index_set(stat)
                if length(∩(index_set, prefix_set)) > 0
                    potential_vars = ∪(potential_vars, index_set)
                end
            end
            potential_vars = setdiff(potential_vars, prefix_set)
            if length(potential_vars) == 0
                # If the query isn't connected, we will need to include a cross product
                potential_vars = setdiff(all_vars, prefix_set)
            end

            for new_var in potential_vars
                new_prefix_set = union(prefix_set, [new_var])
                new_prefix = [prefix..., new_var]
                rf_set = get_reformat_set(all_stats, new_prefix)
                output_compat = get_output_compat(output_vars, new_prefix)
                new_plan_class = (new_prefix_set, output_compat, rf_set)
                new_cost = get_prefix_cost(new_prefix_set, conjunct_stats, disjunct_stats) + cost
                new_plan = (new_prefix, new_cost)

                alt_cost = Inf
                if haskey(new_plans, new_plan_class)
                    alt_cost = new_plans[new_plan_class][2]
                end

                if new_cost <= alt_cost
                    new_plans[new_plan_class] = new_plan
                end
            end
        end

        plans_by_set = Dict()
        for (plan_class, plan) in new_plans
            idx_set = plan_class[1]
            if !haskey(plans_by_set, idx_set)
                plans_by_set[idx_set] = Dict()
            end
            plans_by_set[idx_set][plan_class] = plan
        end

        # If a plan has worse reformatting & worse cost than another plan, we don't need to
        # consider it further.
        undominated_plans = Dict()
        for (plan_class_1, plan_1) in new_plans
            cost_1 = plan_1[2]
            output_compat_1 = plan_class_1[2]
            reformat_set_1 = plan_class_1[3]
            if cost_1 + cost_of_plan_class(plan_class_1, reformat_costs, output_size) > cost_bound
                continue
            end
            is_dominated = false
            for (plan_class_2, plan_2) in plans_by_set[plan_class_1[1]]
                cost_2 = plan_2[2]
                output_compat_2 = plan_class_2[2]
                reformat_set_2 = plan_class_2[3]
                if cost_1 > cost_2 && output_compat_1 >= output_compat_2 && reformat_set_2 ⊆ reformat_set_1
                    is_dominated = true
                    break
                end
            end
            if !is_dominated
                undominated_plans[plan_class_1] = plan_1
            end
        end

        if !isinf(top_k) && length(undominated_plans) > top_k
            plan_and_cost = [(p[2] + cost_of_plan_class(pc, reformat_costs, output_size), pc=>p) for (pc, p) in undominated_plans]
            sort!(plan_and_cost, by=(x)->x[1])
            undominated_plans = Dict(x[2] for x in plan_and_cost[1:top_k])
        end
        optimal_plans = undominated_plans
    end
    min_cost = Inf
    best_prefix = nothing
    best_plan_class = nothing
    for (plan_class, plan) in optimal_plans
        cur_cost = plan[2] + cost_of_plan_class(plan_class, reformat_costs, output_size)
        if cur_cost <= min_cost
            min_cost = cur_cost
            best_prefix = plan[1]
            best_plan_class = plan_class
        end
    end
    return best_prefix, min_cost
end

GREEDY_PLAN_K = 10

function get_join_loop_order(disjunct_and_conjunct_stats, output_stats::TensorStats, output_order::Union{Nothing, Vector{IndexExpr}})
    if length(union([get_index_set(s) for s in disjunct_and_conjunct_stats.disjuncts]...,
                    [get_index_set(s) for s in disjunct_and_conjunct_stats.conjuncts]...)) == 0
        return IndexExpr[]
    end
    greedy_order, greedy_cost = get_join_loop_order_bounded(disjunct_and_conjunct_stats, output_stats, output_order, Inf, GREEDY_PLAN_K)
    exact_order, exact_cost = get_join_loop_order_bounded(disjunct_and_conjunct_stats, output_stats, output_order,  greedy_cost * 1.1, Inf)

    if exact_cost > greedy_cost
        println("Exact Cost: $exact_cost")
        println("Exact Order: $exact_order")
        println("Greedy Cost: $greedy_cost")
        println("Greedy Order: $greedy_order")
    end
    return exact_order
end
