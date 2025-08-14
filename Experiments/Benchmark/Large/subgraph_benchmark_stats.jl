include("../../Experiments.jl")


datasets = [human, aids, yeast_lite, dblp_lite,youtube_lite]
experiments = ExperimentParams[]
for data in datasets
    push!(experiments, ExperimentParams(workload=data, faq_optimizer=pruned; stats_type=NaiveStats, warm_start=true, description="Uniform Estimator", timeout=600))
    push!(experiments, ExperimentParams(workload=data, faq_optimizer=pruned; stats_type=DCStats, warm_start=true, description="Chain Bound", timeout=600))
end
#run_experiments(experiments)
colors=[palette(:default)[1] palette(:default)[16]]

graph_grouped_box_plot(experiments; y_type=execute_time, grouping=description, y_label="Execute Time", group_order=["Chain Bound", "Uniform Estimator"], filename="stats_mb", color=colors)
