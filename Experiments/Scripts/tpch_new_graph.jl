
data = vcat(CSV.read("Experiments/Results/tpch_inference.csv", DataFrame), 
CSV.read("Experiments/Results/tpch_inference_python.csv", DataFrame), 
CSV.read("Experiments/Results/tpch_inference_polars_serial.csv", DataFrame), 
CSV.read("Experiments/Results/tpch_inference_polars_parallel.csv", DataFrame))
data = data[(data.Method .== "Galley") .|| (data.Method .== "Pandas+Numpy") .|| (data.Method .== "Polars+PyTorch (1 Core)"), :]
data = data[(data.Algorithm .!= "Covariance (SQ)") .&& (data.Algorithm .!= "Covariance (SJ)"), :]
data[(data.Method .== "Polars+PyTorch (1 Core)"), :Method] .= "Polars+PyTorch"
data[!, :RelativeExecTime] = copy(data[!, :ExecuteTime])
for alg in unique(data.Algorithm)
    data[data.Algorithm .== alg, :RelativeExecTime] = data[data.Algorithm .== alg, :RelativeExecTime] ./ data[(data.Algorithm .== alg) .& (data.Method .== "Pandas+Numpy"), :ExecuteTime]
end
data[!, :RelativeExecTime] = log10.(data.RelativeExecTime)
data[!, :RelativeOptTime] = log10.(data.RelativeOptTime)
ordered_algorithms = CategoricalArray(data.Algorithm)
ordered_methods = CategoricalArray(data.Method)
alg_order = ["Covariance (SJ)", "Covariance (SQ)", "Logistic Regression (SJ)", "Logistic Regression (SQ)", "Linear Regression (SJ)", "Linear Regression (SQ)", "Neural Network (SJ)", "Neural Network (SQ)"]
levels!(ordered_algorithms, alg_order)
method_order = ["Galley", "Pandas+Numpy", "Polars+PyTorch"]
levels!(ordered_methods, method_order)
gbplot = StatsPlots.groupedbar(ordered_algorithms,
                    data.RelativeExecTime,
                    group = ordered_methods,
                    legend = :topright,
                    size = (2500/1.5, 1000/1.5),
                    ylabel = "Relative Execution Time",
                    ylims=[-2.1,1.9],
                    yticks=([ -2, -1, 0, 1], [".01", ".1", "1", "10"]),
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
                    color=[palette(:default)[1] palette(:default)[2] palette(:default)[3] palette(:default)[4]  palette(:default)[5]  palette(:default)[7]],
                    fillstyle=[nothing nothing nothing])
hline!([0], color=:grey, lw=2, linestyle=:dash; label="")
savefig(gbplot, "Experiments/Figures/tpch_inference.png")
