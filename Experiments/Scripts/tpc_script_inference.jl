using Finch
using Galley: insert_statistics!, t_undef, fill_table
using Galley
using DataFrames
using CSV
using SparseArrays
using Statistics
using Measures
using CategoricalArrays
using StatsFuns
using StatsPlots
using DuckDB
include("../Experiments.jl")

relu(x) = max(0, x)

sigmoid(x) = 1.0 - exp(x-log1pexp(x))

function simplify_col(df, x, val_to_idx)
    df[!, x] = [val_to_idx[x] for x in df[!, x]]
end

function one_hot_encode_col!(df, x)
    ux = unique(df[!, x])
    transform!(df, @. x => ByRow(isequal(ux)) .=> Symbol(string(x) * '_' * string(ux)))
    select!(df, Not(x))
end

function one_hot_encode_cols!(df, xs)
    for x in xs
        one_hot_encode_col!(df, x)
    end
end

function col_to_join_matrix(df, x, n_rows, n_cols)
    return sparse(1:nrow(df), df[!, x], ones(nrow(df)), n_rows, n_cols)
end

function cols_to_join_tensor(df, xs, dims)
    dims = map(Int64, dims)
    Is = [Int64[] for _ in xs]
    V = Int64[]
    for row in eachrow(df)
        for (i, x) in enumerate(xs)
            push!(Is[i], row[x])
        end
        push!(V, 1)
    end
    tensor = Element(0)
    for _ in 1:length(xs)-1
        tensor = SparseList(tensor)
    end
    tensor = Dense(tensor)
    return Tensor(tensor, fsparse(Is..., V, dims))
end

function align_x_dims(X, x_start, x_dim)
    I, J, V = Int[], Int[], Float64[]
    for i in 1:size(X)[1]
        for j in 1:size(X)[2]
            push!(I, i + x_start)
            push!(J, j)
            push!(V, X[i,j])
        end
    end
    new_X = SparseMatrixCSC(sparse(I, J, V, x_dim, size(X)[2]))
    return Tensor(Dense(SparseList(Element(0.0))), new_X)
end

function finch_lr(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x, θ; dense=false)
    X = dense ? Tensor(Dense(Dense(Element(0.0)))) : Tensor(Dense(Sparse(Element(0.0))))
    P = Tensor(Dense((Element(0.0))))
    f_time = @elapsed @finch begin
        X .= 0
        for i =_
            for o=_
                for s=_
                    for p=_
                        for c=_
                            for j =_
                                X[j, i] += li_tns[p, s, o, i] * order_cust[c, o] * (orders_x[j, o] + customer_x[j, c])
                            end
                        end
                    end
                end
            end
        end
    end
    println("First X Step")
    f_time += @elapsed @finch begin
        for i =_
            for o=_
                for s=_
                    for p=_
                        for j =_
                            X[j, i] += li_tns[p, s, o, i] * (supplier_x[j, s] + part_x[j, p])
                        end
                    end
                end
            end
        end
    end
    println("Built X")
    f_time += @elapsed @finch begin
        P .= 0
        for i=_
            for j=_
                P[i] +=  X[j, i] * θ[j]
            end
        end
    end
    return f_time, P
end

function finch_log(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x, θ; dense=false)
    X = dense ? Tensor(Dense(Dense(Element(0.0)))) : Tensor(Dense(Sparse(Element(0.0))))
    p_inter = Tensor(Dense(Element(0.0)))
    prediction = Tensor(Dense(Element(0.5)))
    f_time = @elapsed @finch begin
        X .= 0
        for i =_
            for o=_
                for s=_
                    for p=_
                        for c=_
                            for j =_
                                X[j, i] += li_tns[p, s, o, i] * order_cust[c, o] * (orders_x[j, o] + customer_x[j, c])
                            end
                        end
                    end
                end
            end
        end
    end
    println("First X Step")
    f_time += @elapsed @finch begin
        for i =_
            for o=_
                for s=_
                    for p=_
                        for j =_
                            X[j, i] += li_tns[p, s, o, i] * (supplier_x[j, s] + part_x[j, p])
                        end
                    end
                end
            end
        end
    end
    println("Built X")
    f_time += @elapsed @finch begin
        p_inter .= 0
        for i=_
            for j=_
                p_inter[i] += X[j, i] * θ[j]
            end
        end

        prediction .= 0.5
        for i = _
            prediction[i] = sigmoid(p_inter[i])
        end
    end
    return f_time, prediction
end

function finch_nn(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x, W1, W2, W3; dense=false)
    X = dense ? Tensor(Dense(Dense(Element(0.0)))) : Tensor(Dense(Sparse(Element(0.0))))
    h1 = Tensor(Dense(Dense(Element(0.0))))
    h1_relu = Tensor(Dense(Dense(Element(0.0))))
    h2 = Tensor(Dense(Dense(Element(0.0))))
    h2_relu = Tensor(Dense(Dense(Element(0.0))))
    h3 = Tensor(Dense(Element(0.0)))
    prediction = Tensor(Dense(Element(0.5)))
    f_time = @elapsed @finch begin
        X .= 0
        for i =_
            for o=_
                for s=_
                    for p=_
                        for c=_
                            for j =_
                                X[j, i] += li_tns[p, s, o, i] * order_cust[c, o] * (orders_x[j, o] + customer_x[j, c])
                            end
                        end
                    end
                end
            end
        end
    end
    println("First X Step")
    f_time += @elapsed @finch begin
        for i =_
            for o=_
                for s=_
                    for p=_
                        for j =_
                            X[j, i] += li_tns[p, s, o, i] * (supplier_x[j, s] + part_x[j, p])
                        end
                    end
                end
            end
        end
    end
    println("Built X")
    f_time += @elapsed @finch begin
        h1 .= 0
        for i=_
            for j=_
                for k=_
                    h1[k, i] += X[j, i] * W1[k, j]
                end
            end
        end
    end

    f_time += @elapsed @finch begin
        h1_relu .= 0
        for i=_
            for k=_
                h1_relu[k, i] = max(0, h1[k, i])
            end
        end
    end

    f_time += @elapsed @finch begin
        h2 .= 0
        for i=_
            for j=_
                for k=_
                    h2[k, i] += h1_relu[j, i] * W2[k, j]
                end
            end
        end
    end

    f_time += @elapsed @finch begin
        h2_relu .= 0
        for i=_
            for k=_
                h2_relu[k, i] = max(0, h2[k, i])
            end
        end
    end

    f_time += @elapsed @finch begin
        h3 .= 0
        for i=_
            for j=_
                h3[i] += h2_relu[j, i] * W3[j]
            end
        end
    end

    f_time += @elapsed @finch begin
        prediction .= 0.5
        for i = _
            prediction[i] = sigmoid(h3[i])
        end
    end
    return f_time, prediction
end

function finch_cov(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x; dense=false)
    X = dense ? Tensor(Dense(Dense(Element(0.0)))) : Tensor(Dense(Sparse(Element(0.0))))
    cov = Tensor(Dense(Dense(Element(0.0))))
    f_time = @elapsed @finch begin
        X .= 0
        for i =_
            for o=_
                for s=_
                    for p=_
                        for c=_
                            for j =_
                                X[j, i] += li_tns[p, s, o, i] * order_cust[c, o] * (orders_x[j, o] + customer_x[j, c])
                            end
                        end
                    end
                end
            end
        end
    end
    println("First X Step")
    f_time += @elapsed @finch begin
        for i =_
            for o=_
                for s=_
                    for p=_
                        for j =_
                            X[j, i] += li_tns[p, s, o, i] * (supplier_x[j, s] + part_x[j, p])
                        end
                    end
                end
            end
        end
    end
    println("Built X")
    f_time += @elapsed @finch begin
        cov .= 0
        for i=_
            for j=_
                for k=_
                    cov[k, j] += X[j, i] * X[k, i]
                end
            end
        end
    end
    return f_time, cov
end

function finch_lr2(li_tns, supplier_x1, supplier_x2, part_x,  θ; dense=false)
    println("li_tns: $(countstored(li_tns))")
    X = dense ? Tensor(Dense(Sparse(Dense(Element(0.0))))) : Tensor(Dense(Sparse(Sparse(Element(0.0)))))
    P = Tensor(Dense(SparseList(Element(0.0))))
    f_time = @elapsed @finch begin
        X .= 0
        for p = _
            for i1 =_
                for i2 =_
                    for s1 =_
                        for s2 =_
                            for j =_
                                X[j, i2, i1] += li_tns[s1, i1, p] * li_tns[s2, i2, p] * (part_x[j, p] + supplier_x1[j, s1] + supplier_x2[j, s2])
                            end
                        end
                    end
                end
            end
        end
    end
    println("Built X")
    f_time += @elapsed @finch begin
        P .= 0
        for i1=_
            for i2=_
                for j=_
                        P[i2, i1] += X[j, i2, i1] * θ[j]
                end
            end
        end
    end
    println("Made Predictions")
    return f_time, P
end

function finch_log2(li_tns, supplier_x1, supplier_x2, part_x, θ; dense=false)
    X = dense ? Tensor(Dense(Sparse(Dense(Element(0.0))))) : Tensor(Dense(Sparse(Sparse(Element(0.0)))))
    prediction_inter = Tensor(Dense(SparseList(Element(0.0))))
    prediction = Tensor(Dense(SparseList(Element(0.5))))
    f_time = @elapsed @finch begin
        X .= 0
        for p = _
            for i1 =_
                for i2 =_
                    for s1 =_
                        for s2 =_
                            for j =_
                                X[j, i2, i1] += li_tns[s1, i1, p] * li_tns[s2, i2, p] * (part_x[j, p] + supplier_x1[j, s1] + supplier_x2[j, s2])
                            end
                        end
                    end
                end
            end
        end
    end
    println("nnz(X): $(countstored(X))")
    println("size(X): $(size(X))")
    f_time += @elapsed @finch begin
        prediction_inter .= 0
        for i1=_
            for i2=_
                for k=_
                    prediction_inter[i2, i1] += X[k, i2, i1] * θ[k]
                end
            end
        end
    end
    println("Made Predictions")
    println("nnz(prediction): $(countstored(prediction_inter))")
    println("size(prediction): $(size(prediction_inter))")
    f_time += @elapsed @finch begin
        prediction .= 0.5
        for i1=_
            for i2=_
                prediction[i2, i1] = sigmoid(prediction_inter[i2, i1])
            end
        end
    end
    return f_time, prediction
end

function finch_cov2(li_tns, supplier_x1, supplier_x2, part_x; dense=false)
    X = dense ? Tensor(Dense(Sparse(Dense(Element(0.0))))) : Tensor(Dense(Sparse(Sparse(Element(0.0)))))
    cov = Tensor(Dense(Dense(Element(0.0))))
    f_time = @elapsed @finch begin
        X .= 0
        for p = _
            for i1 =_
                for i2 =_
                    for s1 =_
                        for s2 =_
                            for j =_
                                X[j, i2, i1] += li_tns[s1, i1, p] * li_tns[s2, i2, p] * (part_x[j, p] + supplier_x1[j, s1] + supplier_x2[j, s2])
                            end
                        end
                    end
                end
            end
        end
    end
    println("nnz(X): $(countstored(X))")
    println("size(X): $(size(X))")
    f_time += @elapsed @finch begin
        cov .= 0
        for i1=_
            for i2=_
                for k=_
                    for j=_
                        cov[j, k] += X[k, i2, i1] * X[j, i2, i1]
                    end
                end
            end
        end
    end
    return f_time, cov
end

function finch_nn2(li_tns, supplier_x1, supplier_x2, part_x, W1, W2, W3; dense=false)
    X = dense ? Tensor(Dense(Sparse(Dense(Element(0.0))))) : Tensor(Dense(Sparse(Sparse(Element(0.0)))))
    h1 = Tensor(Dense(Sparse((Dense(Element(0.0))))))
    h1_relu = Tensor(Dense(Sparse(Dense(Element(0.0)))))
    h2 = Tensor(Dense(Sparse(Dense(Element(0.0)))))
    h2_relu = Tensor(Dense(Sparse(Dense(Element(0.0)))))
    h3 = Tensor(Dense(Sparse(Element(0.0))))
    prediction = Tensor(Dense(Sparse(Element(0.5))))
    f_time = @elapsed @finch begin
        X .= 0
        for p = _
            for i1 =_
                for i2 =_
                    for s1 =_
                        for s2 =_
                            for j =_
                                X[j, i2, i1] += li_tns[s1, i1, p] * li_tns[s2, i2, p] * (part_x[j, p] + supplier_x1[j, s1] + supplier_x2[j, s2])
                            end
                        end
                    end
                end
            end
        end
    end
    f_time += @elapsed @finch begin
        h1 .= 0
        for i1=_
            for i2=_
                for j=_
                    for k=_
                        h1[k, i2, i1] += X[j, i2, i1] * W1[k, j]
                    end
                end
            end
        end
    end

    f_time += @elapsed @finch begin
        h1_relu .= 0
        for i1=_
            for i2=_
                for k=_
                    h1_relu[k, i2, i1] = max(0, h1[k, i2, i1])
                end
            end
        end
    end

    f_time += @elapsed @finch begin
        h2 .= 0
        for i1=_
            for i2=_
                for j=_
                    for k=_
                        h2[k, i2, i1] += h1_relu[j, i2, i1] * W2[k, j]
                    end
                end
            end
        end
    end

    f_time += @elapsed @finch begin
        h2_relu .= 0
        for i1=_
            for i2=_
                for k=_
                    h2_relu[k, i2, i1] = max(0, h2[k, i2, i1])
                end
            end
        end
    end

    f_time += @elapsed @finch begin
        h3 .= 0
        for i1=_
            for i2=_
                for k=_
                    h3[i2, i1] += h2_relu[k, i2, i1] * W3[k]
                end
            end
        end
    end

    f_time += @elapsed @finch begin
        prediction .= 0.5
        for i1 = _
            for i2=_
                prediction[i2, i1] = sigmoid(h3[i2, i1])
            end
        end
    end
    return f_time, prediction
end

function main()
    # First, we load the TPCH data
    customer = CSV.read("Experiments/Data/TPCH/customer.tbl", DataFrame, delim='|', header=[:CustomerKey, :Name, :Address, :NationKey, :Phone, :AcctBal, :MktSegment, :Comment, :Col9])
    lineitem = CSV.read("Experiments/Data/TPCH/lineitem.tbl", DataFrame, delim='|', header=[:OrderKey, :PartKey, :SuppKey, :LineNumber, :Quantity, :ExtendedPrice, :Discount, :Tax, :ReturnFlag, :LineStatus, :ShipDate, :CommitDate, :ReceiptDate, :ShipInstruct, :ShipMode, :Comment])
    orders = CSV.read("Experiments/Data/TPCH/orders.tbl", DataFrame, delim='|', header=[:OrderKey, :CustomerKey, :OrderStatus, :TotalPrice, :OrderDate, :OrderPriority, :Clerk, :ShipPriority, :Comment])
    partsupp = CSV.read("Experiments/Data/TPCH/partsupp.tbl", DataFrame, delim='|', header=[:PartKey, :SuppKey, :AvailQty, :SupplyCost, :Comment])
    part = CSV.read("Experiments/Data/TPCH/part.tbl", DataFrame, delim='|', header=[:PartKey, :Name, :MFGR, :Brand, :Type, :Size, :Container, :RetailPrice, :Comment])
    supplier = CSV.read("Experiments/Data/TPCH/supplier.tbl", DataFrame, delim='|', header=[:SuppKey, :Name, :Address, :NationKey, :Phone, :AcctBal, :Comment])
    lineitem[!, :LineItemKey] = 1:nrow(lineitem)
    orderkey_idx = Dict(o => i for (i, o) in enumerate(unique(orders.OrderKey)))
    partkey_idx = Dict(p => i for (i, p) in enumerate(unique(part.PartKey)))
    suppkey_idx = Dict(s => i for (i, s) in enumerate(unique(supplier.SuppKey)))
    customerkey_idx = Dict(c => i for (i, c) in enumerate(unique(customer.CustomerKey)))

    simplify_col(lineitem, :OrderKey, orderkey_idx)
    simplify_col(lineitem, :PartKey, partkey_idx)
    simplify_col(lineitem, :SuppKey, suppkey_idx)
    simplify_col(orders, :OrderKey, orderkey_idx)
    simplify_col(orders, :CustomerKey, customerkey_idx)
    simplify_col(customer, :CustomerKey, customerkey_idx)
    simplify_col(part, :PartKey, partkey_idx)
    simplify_col(supplier, :SuppKey, suppkey_idx)

    # Conceptually, each row of X is going to be a line item.
    lineitem = lineitem[!, [:LineItemKey, :OrderKey, :PartKey, :SuppKey, :LineNumber, :Quantity, :ExtendedPrice, :Discount, :Tax, :ReturnFlag, :LineStatus, :ShipMode]]
    one_hot_encode_cols!(lineitem, [:ReturnFlag, :LineStatus, :ShipMode])
    orders = orders[!, [:OrderKey, :CustomerKey, :OrderStatus, :TotalPrice, :OrderPriority, :ShipPriority]]
    orders.CustomerKey = orders.CustomerKey .+ 1
    one_hot_encode_cols!(orders, [:OrderStatus, :OrderPriority, :ShipPriority])
    customer = customer[!, [:CustomerKey, :NationKey, :AcctBal, :MktSegment]]
    one_hot_encode_cols!(customer, [:NationKey, :MktSegment])
    partsupp = partsupp[!, [:PartKey, :SuppKey]]
    supplier = supplier[!, [:SuppKey, :NationKey, :AcctBal]]
    one_hot_encode_cols!(supplier, [:NationKey])
    part = part[!, [:PartKey, :MFGR, :Brand, :Size, :Container, :RetailPrice]]
    one_hot_encode_cols!(part, [:MFGR, :Brand, :Container])
    li_tns = cols_to_join_tensor(lineitem, (:PartKey, :SuppKey, :OrderKey,  :LineItemKey), (maximum(values(partkey_idx)), maximum(values(suppkey_idx)), maximum(values(orderkey_idx)), nrow(lineitem)))
    li_tns2 = cols_to_join_tensor(lineitem, (:SuppKey, :LineItemKey, :PartKey), (maximum(values(suppkey_idx)), nrow(lineitem), maximum(values(partkey_idx))))
    orders_x = floor.(Matrix(select(orders, Not([:OrderKey, :CustomerKey])))') .% 100
    order_cust = cols_to_join_tensor(orders, (:CustomerKey, :OrderKey), (maximum(values(customerkey_idx)), maximum(values(orderkey_idx))))
    customer_x = floor.(Matrix(select(customer, Not([:CustomerKey])))') .% 100
    supplier_x = floor.(Matrix(select(supplier, Not([:SuppKey])))') .% 100
    part_x = floor.(Matrix(select(part, Not([:PartKey])))') .% 100

    x_starts = cumsum([0, size(orders_x)[1], size(customer_x)[1], size(supplier_x)[1], size(part_x)[1]])
    x_dim = x_starts[end]
    orders_x = align_x_dims(orders_x, x_starts[1], x_dim)
    customer_x = align_x_dims(customer_x, x_starts[2], x_dim)
    supplier_x = align_x_dims(supplier_x, x_starts[3], x_dim)
    part_x = align_x_dims(part_x, x_starts[4], x_dim)
    n_reps = 5
    optimizer = pruned
    X_g = Mat(:i, :j, Σ(:o, :s, :p, MapJoin(*,  Input(li_tns, :p, :s, :o, :i, "li_tns"),
                                                MapJoin(+, Input(orders_x, :j, :o, "orders_x"),
                                                            Σ(:c, MapJoin(*, Input(order_cust, :c, :o, "order_cust"),
                                                                             Input(customer_x, :j, :c, "customer_x"))),
                                                            Input(supplier_x, :j, :s, "supplier_x"),
                                                            Input(part_x, :j, :p, "part_x")))))
    θ = Tensor(Dense(Element(0)), ones(Int, size(supplier_x)[1]) .% 100)
 
    # ---------------- Linear Regression On Star Join -------------------
    P_query = Query(:out, Mat(:i, Aggregate(+, :j, MapJoin(*, X_g[:i, :j], Input(θ, :j)))))
    g_times = []
    g_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(P_query, faq_optimizer=optimizer, ST=DCStats,  verbose=0)
        push!(g_times, result_galley.execute_time)
        push!(g_opt_times, result_galley.opt_time)
    end
    result = galley(P_query, faq_optimizer=optimizer, ST=DCStats, verbose=3)

    f_times = []
    f_ls = []
    for _ in 1:n_reps
        f_time, p = finch_lr(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x, θ; dense=true)
        push!(f_times, f_time)
        push!(f_ls, p)
    end
    f_sp_times = []
    f_sp_ls = []
    for _ in 1:n_reps
        f_time, p = finch_lr(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x, θ; dense=false)
        push!(f_sp_times, f_time)
        push!(f_sp_ls, p)
    end

    # ---------------- Logistic Regression On Star Join -------------------
    p_query = Query(:out, Mat(:i, MapJoin(sigmoid, Σ(:k, MapJoin(*, X_g[:i, :k], Input(θ, :k))))))
    g_log_times = []
    g_log_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(p_query, faq_optimizer=optimizer, ST=DCStats, verbose=0)
        push!(g_log_times, result_galley.execute_time)
        push!(g_log_opt_times, result_galley.opt_time)
    end
    result_log = galley(p_query, faq_optimizer=optimizer, ST=DCStats, verbose=3)

    f_log_times = []
    f_logs = []
    for _ in 1:n_reps
        f_time, prediction = finch_log(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x, θ; dense=true)
        push!(f_log_times, f_time)
        push!(f_logs, prediction)
    end

    f_sp_log_times = []
    f_sp_logs = []
    for _ in 1:n_reps
        f_time, prediction = finch_log(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x, θ; dense=false)
        push!(f_sp_log_times, f_time)
        push!(f_sp_logs, prediction)
    end

    # ---------------- Neural Network Inference On Star Join -------------------
    hidden_layer_size = 25
    feature_size = size(part_x)[1]
    W1 = Tensor(Dense(Dense(Element(0.0))), rand(Int, hidden_layer_size, feature_size) .% 10)
    h1 = Mat(:i, :k1, MapJoin(max, 0, Σ(:j, MapJoin(*, X_g[:i, :j], Input(W1, :k1, :j)))))
    W2 = Tensor(Dense(Dense(Element(0.0))), rand(Int, hidden_layer_size, hidden_layer_size) .% 10)
    h2 = Mat(:i, :k2, MapJoin(max, 0,  Σ(:k1, MapJoin(*, h1[:i, :k1], Input(W2, :k2, :k1)))))
    W3 = Tensor(Dense(Element(0.0)), rand(Int, hidden_layer_size) .% 10)
    h3 = Mat(:i, MapJoin(sigmoid, Σ(:k2, MapJoin(*, h2[:i, :k2], Input(W3, :k2)))))
    P = Query(:out, h3)
    g_nn_times = []
    g_nn_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(P, faq_optimizer=optimizer, ST=DCStats,  verbose=0)
        push!(g_nn_times, result_galley.execute_time)
        push!(g_nn_opt_times, result_galley.opt_time)
    end
    result_nn = galley(P, faq_optimizer=optimizer, ST=DCStats, verbose=3)

    f_nn_times = []
    f_nns = []
    for _ in 1:n_reps
        f_time, prediction = finch_nn(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x, W1, W2, W3; dense=true)
        push!(f_nn_times, f_time)
        push!(f_nns, prediction)
    end

    f_sp_nn_times = []
    f_sp_nns = []
    for _ in 1:n_reps
        f_time, prediction = finch_nn(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x, W1, W2, W3; dense=false)
        push!(f_sp_nn_times, f_time)
        push!(f_sp_nns, prediction)
    end

    # ---------------- Covariance Matrix Computation On Star Join -------------------
    P = Query(:out, Mat(:j, :k, Σ(:i, MapJoin(*, X_g[:i, :j], X_g[:i, :k]))))
    g_cov_times = []
    g_cov_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(P, faq_optimizer=optimizer, ST=DCStats,  verbose=0)
        push!(g_cov_times, result_galley.execute_time)
        push!(g_cov_opt_times, result_galley.opt_time)
    end
    result_cov = galley(P, faq_optimizer=optimizer, ST=DCStats, verbose=3)

    f_cov_times = []
    f_covs = []
    for _ in 1:n_reps
        f_time, prediction = finch_cov(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x; dense=true)
        push!(f_cov_times, f_time)
        push!(f_covs, prediction)
    end

    f_sp_cov_times = []
    f_sp_covs = []
    for _ in 1:n_reps
        f_time, prediction = finch_cov(li_tns, orders_x, order_cust, customer_x, supplier_x, part_x; dense=false)
        push!(f_sp_cov_times, f_time)
        push!(f_sp_covs, prediction)
    end

    # TODO: print out the median, minimum and max to get a sense of variance
    println("Galley Exec: (Min: $(minimum(g_times[2:end])), Mean: $(mean(g_times[2:end])), Max: $(maximum(g_times[2:end]))]")
    println("Galley Opt: (Min: $(minimum(g_opt_times[2:end])), Mean: $(mean(g_opt_times[2:end])), Max: $(maximum(g_opt_times[2:end]))]")
    println("Finch (Dense) Exec: $(mean(f_times[2:end]))")
    println("Finch (Sparse) Exec: $(mean(f_sp_times[2:end]))")
    println("∑|F_i - G_i|: $(sum(abs.(f_ls[1] - result.value)))")
    println("Galley Log Exec: (Min: $(minimum(g_log_times[2:end])), Mean: $(mean(g_log_times[2:end])), Max: $(maximum(g_log_times[2:end]))]")
    println("Galley Log Opt: (Min: $(minimum(g_log_opt_times[2:end])), Mean: $(mean(g_log_opt_times[2:end])), Max: $(maximum(g_log_opt_times[2:end]))]")
    println("Finch (Dense) Log Exec: $(mean(f_log_times[2:end]))")
    println("Finch (Sparse) Log Exec: $(mean(f_sp_log_times[2:end]))")
    println("∑|F_i - G_i|: $(sum(abs.(f_logs[1] - result_log.value)))")
    println("Galley NN Exec: (Min: $(minimum(g_nn_times[2:end])), Mean: $(mean(g_nn_times[2:end])), Max: $(maximum(g_nn_times[2:end]))]")
    println("Galley NN Opt: (Min: $(minimum(g_nn_opt_times[2:end])), Mean: $(mean(g_nn_opt_times[2:end])), Max: $(maximum(g_nn_opt_times[2:end]))]")
    println("Finch (Dense) NN Exec: $(mean(f_nn_times[2:end]))")
    println("Finch (Sparse) NN Exec: $(mean(f_sp_nn_times[2:end]))")
    println("∑|F_i - G_i|: $(sum(abs.(f_nns[1] - result_nn.value)))")
    println("Galley Cov Exec: (Min: $(minimum(g_cov_times[2:end])), Mean: $(mean(g_cov_times[2:end])), Max: $(maximum(g_cov_times[2:end]))]")
    println("Galley Cov Opt: (Min: $(minimum(g_cov_opt_times[2:end])), Mean: $(mean(g_cov_opt_times[2:end])), Max: $(maximum(g_cov_opt_times[2:end]))]")
    println("Finch (Dense) Cov Exec: $(mean(f_cov_times[2:end]))")
    println("Finch (Sparse) Cov Exec: $(mean(f_sp_cov_times[2:end]))")
    println("∑|F_i - G_i|: $(sum(abs.(f_covs[1] - result_cov.value)))")

    f_ls = []
    f_sp_ls = []
    f_logs = []
    f_sp_logs = []
    f_nns = []
    results = [("Algorithm", "Method", "ExecuteTime", "OptTime")]
    push!(results, ("Linear Regression (SQ)", "Galley", string(mean(g_times[2:end])), string(mean(g_opt_times[2:end]))))
    push!(results, ("Linear Regression (SQ)", "Finch (Dense)", string(mean(f_times[2:end])), string(0)))
    push!(results, ("Linear Regression (SQ)", "Finch (Sparse)", string(mean(f_sp_times[2:end])), string(0)))
    push!(results, ("Logistic Regression (SQ)", "Galley", string(mean(g_log_times[2:end])), string(mean(g_log_opt_times[2:end]))))
    push!(results, ("Logistic Regression (SQ)", "Finch (Dense)", string(mean(f_log_times[2:end])), string(0)))
    push!(results, ("Logistic Regression (SQ)", "Finch (Sparse)", string(mean(f_sp_log_times[2:end])), string(0)))
    push!(results, ("Neural Network (SQ)", "Galley", string(mean(g_nn_times[2:end])), string(mean(g_nn_opt_times[2:end]))))
    push!(results, ("Neural Network (SQ)", "Finch (Dense)", string(mean(f_nn_times[2:end])), string(0)))
    push!(results, ("Neural Network (SQ)", "Finch (Sparse)", string(mean(f_sp_nn_times[2:end])), string(0)))
    push!(results, ("Covariance (SQ)", "Galley", string(mean(g_cov_times[2:end])), string(mean(g_cov_opt_times[2:end]))))
    push!(results, ("Covariance (SQ)", "Finch (Dense)", string(mean(f_cov_times[2:end])), string(0)))
    push!(results, ("Covariance (SQ)", "Finch (Sparse)", string(mean(f_sp_cov_times[2:end])), string(0)))

 
    # This formulation makes each row of the output a pair of line items for the same part
    # and includes information about their suppliers li_tns[s1, i1, p]
    supplier_x = floor.(Matrix(select(supplier, Not([:SuppKey])))') .% 100
    part_x = floor.(Matrix(select(part, Not([:PartKey])))') .% 100
    x_starts = cumsum([0, size(supplier_x)[1], size(supplier_x)[1], size(part_x)[1]])
    x_dim = x_starts[end]
    supplier_x1 = align_x_dims(supplier_x, x_starts[1], x_dim)
    supplier_x2 = align_x_dims(supplier_x, x_starts[2], x_dim)
    part_x = align_x_dims(part_x, x_starts[3], x_dim)

    X_g = Mat(:i2, :i1, :j,
              Σ(:p, :s1, :s2, MapJoin(*,Input(li_tns2, :s1, :i1, :p, "li_part"),
                                        Input(li_tns2, :s2, :i2, :p, "li_part"),
                                        MapJoin(+,
                                                    Input(part_x, :j, :p, "part_x"),
                                                    Input(supplier_x1, :j, :s1, "supplier_x1"),
                                                    Input(supplier_x2, :j, :s2, "supplier_x2")))))
    θ = Tensor(Dense(Element(0.0)), ones(Int, size(supplier_x1)[1]) .% 100)
 
    # ---------------- Linear Regression On Many-Many Join -------------------
    p_query = Query(:out, Materialize(t_undef, t_undef, :i2, :i1,  Σ(:k, MapJoin(*, X_g[:i2, :i1, :k], Input(θ, :k)))))
    g_times = []
    g_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(p_query, ST=DCStats, faq_optimizer=optimizer,  verbose=0)
        push!(g_times, result_galley.execute_time)
        push!(g_opt_times, result_galley.opt_time)
    end
    result = galley(p_query, ST=DCStats, faq_optimizer=optimizer, verbose=3)

    f_sp_times = []
    f_l = nothing
    for _ in 1:n_reps
        f_time, P = finch_lr2(li_tns2, supplier_x1, supplier_x2, part_x, θ)
        push!(f_sp_times, f_time)
        f_l = P
    end

    f_times = []
    f_l = nothing
    for _ in 1:n_reps
        f_time, P = finch_lr2(li_tns2, supplier_x1, supplier_x2, part_x, θ; dense=true)
        push!(f_times, f_time)
        f_l = P
    end

    # ---------------- Logistic Regression On Many-Many Join -------------------
    p_g = Query(:P, Materialize(t_undef, t_undef, :i2, :i1, MapJoin(sigmoid, Σ(:k, MapJoin(*, X_g[:i2, :i1, :k], Input(θ, :k))))))
    g_log_times = []
    g_log_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(p_g, ST=DCStats, faq_optimizer=optimizer, verbose=0)
        push!(g_log_times, result_galley.execute_time)
        push!(g_log_opt_times, result_galley.opt_time)
    end
    result_log = galley(p_g, ST=DCStats, faq_optimizer=optimizer, verbose=3)

    f_sp_log_times = []
    f_log = nothing
    for _ in 1:n_reps
        f_time, P = finch_log2(li_tns2, supplier_x1, supplier_x2, part_x, θ)
        push!(f_sp_log_times, f_time)
        f_log = P
    end

    f_log_times = []
    f_log = nothing
    for _ in 1:n_reps
        f_time, P = finch_log2(li_tns2, supplier_x1, supplier_x2, part_x, θ; dense=true)
        push!(f_log_times, f_time)
        f_log = P
    end

    # ---------------- Covariance Matrix Computation On Many-Many Join -------------------
    cov_g = Query(:P, Materialize(t_undef, t_undef, :j, :k, Σ(:i2, :i1, MapJoin(*, X_g[:i2, :i1, :k], X_g[:i2, :i1, :j]))))
    g_cov_times = []
    g_cov_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(cov_g, ST=DCStats, faq_optimizer=optimizer, verbose=0)
        push!(g_cov_times, result_galley.execute_time)
        push!(g_cov_opt_times, result_galley.opt_time)
    end
    result_cov = galley(cov_g, ST=DCStats, faq_optimizer=optimizer, verbose=3)

    f_sp_cov_times = []
    f_cov = nothing
    for _ in 1:n_reps
        f_time, P = finch_cov2(li_tns2, supplier_x1, supplier_x2, part_x)
        push!(f_sp_cov_times, f_time)
        f_cov = P
    end

    f_cov_times = []
    f_cov = nothing
    for _ in 1:n_reps
        f_time, P = finch_cov2(li_tns2, supplier_x1, supplier_x2, part_x; dense=true)
        push!(f_cov_times, f_time)
        f_cov = P
    end

    # ---------------- Neural Network Inference On Self Join -------------------
    hidden_layer_size = 25
    feature_size = size(part_x)[1]
    W1 = Tensor(Dense(Dense(Element(0.0))), rand(Int, hidden_layer_size, feature_size) .% 10)
    h1 = Mat(:i2, :i1, :k1, MapJoin(max, 0, Σ(:j, MapJoin(*, X_g[:i2, :i1, :j], Input(W1, :k1, :j)))))
    W2 = Tensor(Dense(Dense(Element(0.0))), rand(Int, hidden_layer_size, hidden_layer_size) .% 10)
    h2 = Mat(:i2, :i1, :k2, MapJoin(max, 0, Σ(:k1, MapJoin(*, h1[:i2, :i1, :k1], Input(W2, :k2, :k1)))))
    W3 = Tensor(Dense(Element(0.0)), rand(Int, hidden_layer_size) .% 10)
    h3 = Mat(:i2, :i1, MapJoin(sigmoid, Σ(:k2, MapJoin(*, h2[:i2, :i1, :k2], Input(W3, :k2)))))
    P = Query(:out, h3)
    g_nn_times = []
    g_nn_opt_times = []
    for _ in 1:n_reps
        result_galley = galley(P, faq_optimizer=optimizer, ST=DCStats,  verbose=0)
        push!(g_nn_times, result_galley.execute_time)
        push!(g_nn_opt_times, result_galley.opt_time)
    end
    result_nn = galley(P, faq_optimizer=optimizer, ST=DCStats, verbose=3)

    f_sp_nn_times = []
    f_nn = nothing
    for _ in 1:n_reps
        f_time, P = finch_nn2(li_tns2, supplier_x1, supplier_x2, part_x, W1, W2, W3)
        push!(f_sp_nn_times, f_time)
        f_nn = P
    end


    f_nn_times = []
    f_nn = nothing
    for _ in 1:n_reps
        f_time, P = finch_nn2(li_tns2, supplier_x1, supplier_x2, part_x, W1, W2, W3; dense=true)
        push!(f_nn_times, f_time)
        f_nn = P
    end
    
    println("Galley Exec: (Min: $(minimum(g_times[2:end])), Mean: $(mean(g_times[2:end])), Max: $(maximum(g_times[2:end]))]")
    println("Galley Opt: (Min: $(minimum(g_opt_times[2:end])), Mean: $(mean(g_opt_times[2:end])), Max: $(maximum(g_opt_times[2:end]))]")
    println("Finch (Sparse) Exec: $(mean(f_sp_times[2:end]))")
    println("Finch (Dense) Exec: $(mean(f_times[2:end]))")
    println("F = G: $(sum(abs.(f_l .- result.value)))")
    println("Galley Log Exec: (Min: $(minimum(g_log_times[2:end])), Mean: $(mean(g_log_times[2:end])), Max: $(maximum(g_log_times[2:end]))]")
    println("Galley Log Opt: (Min: $(minimum(g_log_opt_times[2:end])), Mean: $(mean(g_log_opt_times[2:end])), Max: $(maximum(g_log_opt_times[2:end]))]")
    println("Finch (Sparse) Log Exec: $(mean(f_sp_log_times[2:end]))")
    println("Finch (Dense) Log Exec: $(mean(f_log_times[2:end]))")
    println("F = G: $(sum(abs.(f_log .- result_log.value)))")
    println("Galley Cov Exec: (Min: $(minimum(g_cov_times[2:end])), Mean: $(mean(g_cov_times[2:end])), Max: $(maximum(g_cov_times[2:end]))]")
    println("Galley Cov Opt: (Min: $(minimum(g_cov_opt_times[2:end])), Mean: $(mean(g_cov_opt_times[2:end])), Max: $(maximum(g_cov_opt_times[2:end]))]")
    println("Finch (Sparse) Cov Exec: $(mean(f_sp_cov_times[2:end]))")
    println("Finch (Dense) Cov Exec: $(mean(f_cov_times[2:end]))")
    println("F = G: $(sum(abs.(f_cov .- result_cov.value)))")
    println("Galley NN Exec: (Min: $(minimum(g_nn_times[2:end])), Mean: $(mean(g_nn_times[2:end])), Max: $(maximum(g_nn_times[2:end]))]")
    println("Galley NN Opt: (Min: $(minimum(g_nn_opt_times[2:end])), Mean: $(mean(g_nn_opt_times[2:end])), Max: $(maximum(g_nn_opt_times[2:end]))]")
    println("Finch (Sparse) NN Exec: $(mean(f_sp_nn_times[2:end]))")
    println("Finch (Dense) NN Exec: $(mean(f_nn_times[2:end]))")
    println("F = G: $(sum(abs.(f_nn .- result_nn.value)))")

    push!(results, ("Linear Regression (SJ)", "Galley", string(mean(g_times[2:end])), string(mean(g_opt_times[2:end]))))
    push!(results, ("Linear Regression (SJ)", "Finch (Sparse)", string(mean(f_sp_times[2:end])), string(0)))
    push!(results, ("Linear Regression (SJ)", "Finch (Dense)", string(mean(f_times[2:end])), string(0)))
    push!(results, ("Logistic Regression (SJ)", "Galley", string(mean(g_log_times[2:end])), string(mean(g_log_opt_times[2:end]))))
    push!(results, ("Logistic Regression (SJ)", "Finch (Sparse)", string(mean(f_sp_log_times[2:end])), string(0)))
    push!(results, ("Logistic Regression (SJ)", "Finch (Dense)", string(mean(f_log_times[2:end])), string(0)))
    push!(results, ("Covariance (SJ)", "Galley", string(mean(g_cov_times[2:end])), string(mean(g_cov_opt_times[2:end]))))
    push!(results, ("Covariance (SJ)", "Finch (Sparse)", string(mean(f_sp_cov_times[2:end])), string(0)))
    push!(results, ("Covariance (SJ)", "Finch (Dense)", string(mean(f_cov_times[2:end])), string(0)))
    push!(results, ("Neural Network (SJ)", "Galley", string(mean(g_nn_times[2:end])), string(mean(g_nn_opt_times[2:end]))))
    push!(results, ("Neural Network (SJ)", "Finch (Sparse)", string(mean(f_sp_nn_times[2:end])), string(0)))
    push!(results, ("Neural Network (SJ)", "Finch (Dense)", string(mean(f_nn_times[2:end])), string(0)))

    writedlm("Experiments/Results/tpch_inference.csv", results, ',')
    data = vcat(CSV.read("Experiments/Results/tpch_inference.csv", DataFrame), 
                CSV.read("Experiments/Results/tpch_inference_python.csv", DataFrame), 
                CSV.read("Experiments/Results/tpch_inference_polars_serial.csv", DataFrame), 
                CSV.read("Experiments/Results/tpch_inference_polars_parallel.csv", DataFrame))
    data[(data.Method .== "Galley"), :Method] .= "Galley (Opt)"
    data[!, :RelativeOptTime] = copy(data[!, :ExecuteTime]) .+ copy(data[!, :OptTime])
    data[!, :RelativeExecTime] = copy(data[!, :ExecuteTime])
    for alg in unique(data.Algorithm)
        data[data.Algorithm .== alg, :RelativeExecTime] = data[data.Algorithm .== alg, :RelativeExecTime] ./ data[(data.Algorithm .== alg) .& (data.Method .== "Finch (Sparse)"), :ExecuteTime]
        data[data.Algorithm .== alg, :RelativeOptTime] = data[data.Algorithm .== alg, :RelativeOptTime] ./ data[(data.Algorithm .== alg) .& (data.Method .== "Finch (Sparse)"), :ExecuteTime]
    end
    data[!, :RelativeExecTime] = log10.(data.RelativeExecTime)
    data[!, :RelativeOptTime] = log10.(data.RelativeOptTime)
    ordered_algorithms = CategoricalArray(data.Algorithm)
    ordered_methods = CategoricalArray(data.Method)
    alg_order = ["Covariance (SJ)", "Covariance (SQ)", "Logistic Regression (SJ)", "Logistic Regression (SQ)", "Linear Regression (SJ)", "Linear Regression (SQ)", "Neural Network (SJ)", "Neural Network (SQ)"]
    levels!(ordered_algorithms, alg_order)
    method_order = ["Galley (Opt)", "Finch (Dense)", "Finch (Sparse)", "Pandas+Numpy", "Polars+PyTorch (1 Core)", "Polars+PyTorch (24 Core)"]
    levels!(ordered_methods, method_order)
    gbplot = StatsPlots.groupedbar(ordered_algorithms,
                                    data.RelativeOptTime,
                                    group = ordered_methods,
                                    legend = :topright,
                                    size = (3000, 1000),
                                    ylabel = "Relative Runtime",
                                    ylims=[-3.1,1.5],
                                    yticks=([-3, -2, -1, 0, 1], [".001", ".01", ".1", "1", "10"]),
                                    xtickfontsize=22,
                                    ytickfontsize=22,
                                    xrotation=25,
                                    xguidefontsize=20,
                                    yguidefontsize=24,
                                    legendfontsize=22,
                                    left_margin=20Measures.mm,
                                    bottom_margin=35Measures.mm,
                                    fillrange=-4,
                                    legend_columns=2,
                                    color=[palette(:blues)[1] palette(:default)[2] palette(:default)[3] palette(:default)[4]  palette(:default)[5]  palette(:default)[7]],
                                    fillstyle=[nothing nothing nothing nothing nothing nothing])
    n_groups = length(unique(data.Algorithm))
    n_methods = length(unique(ordered_methods))
    left_edges = [i for i in 0:n_groups-1]
    first_pos = left_edges .+ 1/n_methods
    galley_data = data[data.Method .== "Galley (Opt)", :]
    galley_data = collect(zip(galley_data.Algorithm, galley_data.RelativeExecTime))
    sort!(galley_data, by = (x)->[i for (i, alg) in enumerate(alg_order) if alg == x[1]][1])
    exec_time = [x[2] for x in galley_data]
    bar!(gbplot, first_pos, exec_time, bar_width=0.8/n_methods, fillrange=-4, label="Galley (Exec)", color=palette(:default)[1])
    hline!([0], color=:grey, lw=2, linestyle=:dash; label="")
    savefig(gbplot, "Experiments/Figures/tpch_inference.png")
end

main()
