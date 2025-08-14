using Galley
using Finch
using DataFrames
using CSV

function main()
    N = 2000
    A_Sparsity = .1
    B_Sparsity = .1
    densities = [.1, .01, .001, .0001, .00001]
#    densities = [ .001, ]
    forward_times = []
    backward_times =[]
    sum_times = []
    elementwise_times = []
    dense_times = []
    A = Tensor(Dense(SparseList(Element(0.0))), fsprand(N, N, A_Sparsity))
    B = Tensor(Dense(SparseList(Element(0.0))), fsprand(N, N, B_Sparsity))
    A_dense = Tensor(Dense(Dense(Element(0.0))), rand(N, N))
    B_dense = Tensor(Dense(Dense(Element(0.0))), rand(N, N))
    C_dense = Tensor(Dense(Dense(Element(0.0))), rand(N, Int(floor(N/400))))
    for d in densities
        C = Tensor(Dense(SparseList(Element(0.0))), fsprand(N, N, d))
        n_reps = 7
        avg_time_forward = 0
        avg_time_backward = 0
        avg_time_sum = 0
        avg_time_elementwise = 0
        avg_time_dense = 0
        for i in 1:n_reps
            verbosity = i == 2 ? 3 : 0
            if i == 3
                avg_time_forward = 0
                avg_time_backward = 0
                avg_time_sum = 0
                avg_time_elementwise = 0
                avg_time_dense = 0
            end

            E1 = Query(:E1, Mat(:i, :l, Σ(:j, :k, MapJoin(*, Input(A, :i, :j), Input(B, :j, :k), Input(C, :k, :l)))))
            start_time = time()
            galley(E1, verbose=verbosity, faq_optimizer=pruned)
            end_time = time()
            avg_time_forward += end_time - start_time

            E2 = Query(:E2,  Mat(:i, :l, Σ(:j, :k, MapJoin(*, Input(C, :i, :j), Input(B, :j, :k), Input(A, :k, :l)))))
            start_time = time()
            galley(E2, verbose=verbosity, faq_optimizer=pruned)
            end_time = time()
            avg_time_backward += end_time - start_time

            E3 = Query(:E3,  Mat(Σ(:i, :j, :k, :l, MapJoin(*, Input(C, :i, :j), Input(B, :j, :k), Input(A, :k, :l)))))
            start_time = time()
            galley(E3, verbose=verbosity, faq_optimizer=pruned)
            end_time = time()
            avg_time_sum += end_time - start_time

            E4 = Query(:E4,  Mat(:i, :j, MapJoin(*, Input(A_dense, :i, :j), Input(B_dense, :i, :j), Input(C, :i, :j))))
            start_time = time()
            galley(E4, verbose=verbosity, faq_optimizer=pruned)
            end_time = time()
            avg_time_elementwise += end_time - start_time

            # This example uses a non-square matrix to compare dense matmul chains.
            E5 = Query(:E5,  Mat(:i, :l, Σ(:j, :k, MapJoin(*, Input(A_dense, :i, :j), Input(B_dense, :j, :k), Input(C_dense, :k, :l)))))
            start_time = time()
            galley(E5, verbose=verbosity, faq_optimizer=pruned)
            end_time = time()
            avg_time_dense += end_time - start_time
        end
        avg_time_forward /= n_reps - 2
        avg_time_backward /= n_reps - 2
        avg_time_sum /= n_reps - 2
        avg_time_elementwise /= n_reps - 2
        avg_time_dense /= n_reps - 2
        push!(forward_times, avg_time_forward)
        push!(backward_times, avg_time_backward)
        push!(sum_times, avg_time_sum)
        push!(elementwise_times, avg_time_elementwise)
        push!(dense_times, avg_time_dense)
        println("Sparsity: ", d)
        println("Forward Time: ", avg_time_forward)
        println("Backward Time: ", avg_time_backward)
        println("Sum Time: ", avg_time_sum)
        println("Elementwise Time: ", avg_time_elementwise)
        println("Dense (Non-Square) Time: ", avg_time_dense)
    end
    println(forward_times)
    println(backward_times)
    println(sum_times)
    println(elementwise_times)
    println(dense_times)
    data = []
    for i in eachindex(densities)
        push!(data, (Method="Galley (1 Core)", Algorithm="ABC", Sparsity=densities[i], Runtime=forward_times[i]))
        push!(data, (Method="Galley (1 Core)", Algorithm="CBA", Sparsity=densities[i], Runtime=backward_times[i]))
        push!(data, (Method="Galley (1 Core)", Algorithm="SUM(ABC)", Sparsity=densities[i], Runtime=sum_times[i]))
        push!(data, (Method="Galley (1 Core)", Algorithm="A*B*C", Sparsity=densities[i], Runtime=elementwise_times[i]))
        push!(data, (Method="Galley (1 Core)", Algorithm="ABC Rectangular", Sparsity=densities[i], Runtime=elementwise_times[i]))
    end
    CSV.write("Experiments/Results/mat_exps_galley.csv", DataFrame(data))
end

main()


using Plots
using StatsPlots
using DataFrames
using CSV

data = DataFrame(CSV.File("Experiments/Results/mat_exps_galley.csv"))
append!(data, DataFrame(CSV.File("Experiments/Results/mat_exps_pytorch_serial.csv")), promote=true)
append!(data, DataFrame(CSV.File("Experiments/Results/mat_exps_pytorch_parallel.csv")), promote=true)

abc_plt = @df data[data.Algorithm .=="ABC", :] bar([i%5 + floor(i/5)/4 for i in 0:17], :Runtime, group=:Method,
                                        xticks=([.25, 1.25, 2.25, 3.25, 4.25], [".1", ".01", ".001", ".0001", ".00001"]),
                                        yticks=[10^-3, 10^-2, .1, 1, 10, 100],
                                        yscale=:log10, fillrange = 10^(-3), ylims=[10^(-3), 20],
                                        xflip=false, lw=2, xtickfontsize=13,  bar_width=.25, xguidefontsize=15, yguidefontsize=15,
                                        ytickfontsize=14, legendfontsize=14, legend=:topright,
                                        title="ABC",
                                        xlabel="Sparsity of C", ylabel="Execution Time (s)")
savefig(abc_plt, "Experiments/Figures/mat_chain_abc.png")


cba_plt = @df data[data.Algorithm .=="CBA", :] bar([i%5 + floor(i/5)/4 for i in 0:17], :Runtime, group=:Method,
                                        xticks=([.25, 1.25, 2.25, 3.25, 4.25], [".1", ".01", ".001", ".0001", ".00001"]),
                                        yticks=[10^-3, 10^-2, .1, 1, 10, 100],
                                        yscale=:log10, fillrange = 10^(-3), ylims=[10^(-3), 20],
                                        xflip=false, lw=2, xtickfontsize=13,  bar_width=.25, xguidefontsize=15, yguidefontsize=15,
                                        ytickfontsize=14, legendfontsize=14, legend=:topright,
                                        title="CBA",
                                        xlabel="Sparsity of C", ylabel="Execution Time (s)")
savefig(cba_plt, "Experiments/Figures/mat_chain_cba.png")

sum_plt = @df data[data.Algorithm .=="SUM(ABC)", :] bar([i%5 + floor(i/5)/4 for i in 0:17], :Runtime, group=:Method,
                                        xticks=([.25, 1.25, 2.25, 3.25, 4.25], [".1", ".01", ".001", ".0001", ".00001"]),
                                        yticks=[10^-3, 10^-2, .1, 1, 10, 100],
                                        yscale=:log10, fillrange = 10^(-3), ylims=[10^(-3), 20],
                                        xflip=false, lw=2, xtickfontsize=13,  bar_width=.25, xguidefontsize=15, yguidefontsize=15,
                                        ytickfontsize=14, legendfontsize=14, legend=:topright,
                                        title="SUM(ABC)",
                                        xlabel="Sparsity of C", ylabel="Execution Time (s)")
savefig(sum_plt, "Experiments/Figures/mat_chain_sum.png")

sum_plt = @df data[data.Algorithm .=="A*B*C", :] bar([i%5 + floor(i/5)/4 for i in 0:17], :Runtime, group=:Method,
                                        xticks=([.25, 1.25, 2.25, 3.25, 4.25], [".1", ".01", ".001", ".0001", ".00001"]),
                                        yticks=[10^-3, 10^-2, .1, 1, 10],
                                        yscale=:log10, fillrange = 10^(-3), ylims=[10^(-3), 20],
                                        xflip=false, lw=2, xtickfontsize=13,  bar_width=.25, xguidefontsize=15, yguidefontsize=15,
                                        ytickfontsize=14, legendfontsize=14, legend=:topright,
                                        title="A*B*C",
                                        xlabel="Sparsity of C", ylabel="Execution Time (s)")
savefig(sum_plt, "Experiments/Figures/mat_chain_elementwise.png")
