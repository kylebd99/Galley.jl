using SparseArrays
using Finch
using DelimitedFiles
using Distributed
using CSV, DataFrames
using DelimitedFiles: writedlm
using StatsPlots
using Plots
using DuckDB
using Galley
using Galley: FAQ_OPTIMIZERS, relabel_input, reindex_stats, fill_table
include("experiment_params.jl")
include("subgraph_workload.jl")
include("load_workload.jl")
include("subgraph_sql.jl")


function clear_channel(c)
    while isready(c)
        take!(c)
    end
end

function attempt_experiment(experiment::ExperimentParams, starting_query, results_channel, status_channel)
    println("Starting Worker Experiment")
    dbconn = experiment.use_duckdb ? DBInterface.connect(DuckDB.DB, ":memory:") : ( experiment.use_umbra ? "Umbra" : nothing)
    queries = load_workload(experiment.workload, experiment.stats_type, dbconn)
#    queries = [q for q in queries if occursin("sparse_8_120", q.query_path)]
    num_attempted, num_completed, num_correct, num_with_values, _  = (0, 0, 0, 0, false)
    if isready(status_channel)
        num_attempted, num_completed, num_correct, num_with_values, _ = take!(status_channel)
    end
    put!(status_channel, (num_attempted, num_completed, num_correct, num_with_values, false))
    for query in queries[starting_query:end]
        if false && !occursin("sparse_8_120", query.query_path)
            continue
        end
        println("Query Path: ", query.query_path)
        num_attempted +=1
        try
            if experiment.use_umbra
                if query.query_path ∈ [""] # This query kills the umbra server due to an internal error.
                    put!(results_channel, (string(experiment.workload), query.query_type, query.query_path, "0.0", "0.0", "0.0", "0.0", string(true)))
                else
                    result, execute_time, opt_time, compile_time = execute_galley_query_umbra(query.query[1], experiment.use_umbra_parallel)
                    put!(results_channel, (string(experiment.workload), query.query_type, query.query_path, string(execute_time), string(opt_time), string(compile_time), string(result[1]), string(false)))
                    if !isnothing(query.expected_result)
                        if result[1] == query.expected_result
                            num_correct += 1
                        end
                        num_with_values += 1
                    end
                    num_completed += 1
                end
            elseif experiment.use_duckdb
                result = galley(query.query, ST=experiment.stats_type; faq_optimizer = experiment.faq_optimizer, dbconn=dbconn, verbose=0)
                result = galley(query.query, ST=experiment.stats_type; faq_optimizer = experiment.faq_optimizer, dbconn=dbconn, verbose=0)
                if result == "failed"
                    put!(results_channel, (string(experiment.workload), query.query_type, query.query_path, "0.0", "0.0", "0.0", string(true)))
                else
                    put!(results_channel, (string(experiment.workload), query.query_type, query.query_path, string(result.execute_time), string(result.opt_time), "0.0", string(result.value), string(false)))
                    if !isnothing(query.expected_result)
                        if result.value[1] == query.expected_result
                            num_correct += 1
                        end
                        num_with_values += 1
                    end
                    num_completed += 1
                end
            else
                warm_start_time = 0
                if experiment.warm_start
                    println("Warm Start Query Path: ", query.query_path)
                    warm_start_time = @elapsed galley(query.query, ST=experiment.stats_type; faq_optimizer = experiment.faq_optimizer, update_cards=experiment.update_cards, simple_cse=experiment.simple_cse, max_kernel_size=experiment.max_kernel_size, verbose=3)
                    println("Warm Start Time: $warm_start_time")
                end
                result = galley(query.query, ST=experiment.stats_type; faq_optimizer = experiment.faq_optimizer, update_cards=experiment.update_cards, simple_cse=experiment.simple_cse, max_kernel_size=experiment.max_kernel_size, verbose=0)
                println(result)
                if result == "failed"
                    put!(results_channel, (string(experiment.workload), query.query_type, query.query_path, "0.0", "0.0", "0.0", "0.0", string(true)))
                else
                    put!(results_channel, (string(experiment.workload), query.query_type, query.query_path, string(result.execute_time), string(result.opt_time), string(warm_start_time - result.execute_time - result.opt_time), string(result.value), string(false)))
                    if !isnothing(query.expected_result)
                        if all(result.value[1] .== query.expected_result)
                            num_correct += 1
                        else
                            println("Query Incorrect: $(query.query_path) Expected: $(query.expected_result) Returned: $(result.value)")
                            throw(Base.error("WRONG RESULT"))
                        end
                        num_with_values += 1
                    end
                    num_completed += 1
                end
            end

            clear_channel(status_channel)
            put!(status_channel, (num_attempted, num_completed, num_correct, num_with_values, false))
        catch e
            println("Error Occurred: ", e)
            throw(e)
        end
    end
    clear_channel(status_channel)
    println("PUBLISHING FINISH STATUS")
    put!(status_channel, (num_attempted, num_completed, num_correct, num_with_values, true))
end
