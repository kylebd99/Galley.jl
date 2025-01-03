include("../../Experiments.jl")

datasets = [human, aids, yeast_lite, dblp_lite, youtube_lite]
datasets = [youtube_lite]
experiments = ExperimentParams[]
for data in datasets
    push!(experiments, ExperimentParams(workload=data, use_umbra=true, use_umbra_parallel=false, warm_start=true, description="Umbra (Serial)", timeout=600))
    push!(experiments, ExperimentParams(workload=data, use_umbra=true, use_umbra_parallel=true, warm_start=true, description="Umbra (Parallel)", timeout=600))
#    push!(experiments, ExperimentParams(workload=data, faq_optimizer=pruned; stats_type=DCStats, warm_start=true, description="Galley", timeout=600))
#    push!(experiments, ExperimentParams(workload=data, faq_optimizer=greedy; stats_type=DCStats, warm_start=true, description="Galley (Greedy)", timeout=600))
#    push!(experiments, ExperimentParams(workload=data, faq_optimizer=pruned; stats_type=DCStats,  use_duckdb=true, description="Galley + DuckDB Backend", timeout=600))
#    push!(experiments, ExperimentParams(workload=data, faq_optimizer=naive; stats_type=NaiveStats, max_kernel_size=4, use_duckdb=true, description="DuckDB", timeout=600))
end

run_experiments(experiments; use_new_processes=true)
colors = [palette(:default)[1] palette(:default)[6] palette(:default)[5] palette(:default)[7]]
filename = "1subgraph_counting_"
graph_grouped_box_plot(experiments; y_type=overall_time, y_lims=[10^-3, 10^3], grouping=description, group_order=["Galley", "Galley (Greedy)", "Galley + DuckDB Backend", "DuckDB", "Umbra (Serial)", "Umbra (Parallel)"], filename="$(filename)overall", y_label="Execute + Optimize Time (s)", color=colors)
graph_grouped_bar_plot(experiments; y_type=opt_time, y_lims=[10^-3.2, 5], grouping=description, group_order=["Galley", "Galley (Greedy)", "Galley + DuckDB Backend", "DuckDB", "Umbra (Serial)", "Umbra (Parallel)"],  filename="$(filename)opt", y_label="Mean Optimization Time (s)", color=colors)
graph_grouped_box_plot(experiments; y_type=execute_time, y_lims=[10^-4, 10^3], grouping=description, group_order=["Galley", "Galley (Greedy)", "Galley + DuckDB Backend", "DuckDB", "Umbra (Serial)", "Umbra (Parallel)"],  filename="$(filename)execute", y_label="Execution Time (s)", color=colors)
graph_grouped_bar_plot(experiments; y_type=compile_time, y_lims=[10^-2, 10^2], grouping=description, group_order=["Galley", "Galley (Greedy)", "Galley + DuckDB Backend", "DuckDB", "Umbra (Serial)", "Umbra (Parallel)"],  filename="$(filename)compile", y_label="Mean Compile Time (s)", color=colors)
