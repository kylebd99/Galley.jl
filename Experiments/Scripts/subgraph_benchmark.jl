include("../Experiments.jl")


#datasets = instances(SUBGRAPH_DATASET)
datasets = [aids]

experiments = ExperimentParams[]
for dataset in datasets
    push!(experiments, ExperimentParams(workload=dataset, faq_optimizer=hypertree))
end

run_experiments(experiments)

graph_grouped_box_plot(experiments)