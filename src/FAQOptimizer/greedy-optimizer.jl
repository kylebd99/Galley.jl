function greedy_query_to_plan(input_query::PlanNode, ST, use_dnf)
    aq = AnnotatedQuery(input_query, ST, use_dnf)
    queries = []
    cost_cache = Dict()
    total_cost = 0
    reducible_idxs = get_reducible_idxs(aq)
    while !isempty(reducible_idxs)
        cheapest_idx = nothing
        cheapest_cost = Inf
        for idx in reducible_idxs
            cost, _ = cost_of_reduce(idx, aq, cost_cache)
            if cost < cheapest_cost
                cheapest_idx = idx
                cheapest_cost = cost
            end
        end
        query = reduce_idx!(cheapest_idx, aq)
        push!(queries, query)
        reducible_idxs = get_reducible_idxs(aq)
        total_cost += cheapest_cost
    end
    remaining_q = get_remaining_query(aq)
    if !isnothing(remaining_q)
        push!(queries, remaining_q)
    end
    last_query = queries[end]
    last_query.expr = Materialize(aq.output_format..., aq.output_order..., last_query.expr)
    last_query.expr.stats = last_query.expr.expr.stats
    return Plan(queries..., Outputs(queries[end].name)), total_cost
end
