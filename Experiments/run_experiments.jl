function load_worker()
    println("Spawning Worker")
    worker_pid = addprocs(1)[1]
    f = @spawnat worker_pid include("Experiments/run_experiments_worker.jl")
    fetch(f)
    return worker_pid
end

function clear_channel(c)
    while isready(c)
        take!(c)
    end
end

const results_channel = RemoteChannel(()->Channel{Any}(10000), 1)
const status_channel = RemoteChannel(()->Channel{Any}(10000), 1)

function run_experiments(experiment_params::Vector{ExperimentParams}; use_new_processes=true)
    for experiment in experiment_params
        clear_channel(results_channel)
        clear_channel(status_channel)
        results = [("Workload", "QueryType", "QueryPath", "Runtime", "OptTime", "CompileTime", "Result", "Failed")]
        num_attempted, num_completed, num_correct, num_with_values, exp_finished = (0, 0, 0, 0, false)
        put!(status_channel, (num_attempted, num_completed, num_correct, num_with_values, exp_finished))
        worker_pid = -1
        cur_query = 1
        f = nothing
        while !exp_finished
            if use_new_processes
                if worker_pid == -1
                    worker_pid = load_worker()
                end
                f = @spawnat worker_pid attempt_experiment(experiment, cur_query, results_channel, status_channel)
            else
                f = attempt_experiment(experiment, cur_query, results_channel, status_channel)
            end
            load_start = time()
            finished = false
            last_result = time()
            while !finished
                sleep(.01)
                if isready(results_channel)
                    push!(results, take!(results_channel))
                    cur_query += 1
                    last_result = time()
                end
                if isready(status_channel)
                    num_attempted, num_completed, num_correct, num_with_values, exp_finished = fetch(status_channel)
                    finished = exp_finished
                end
                if ((time()-last_result) > experiment.timeout) && (time() - load_start > 200)
                    println("REMOVING WORKER")
                    interrupt(worker_pid)
                    rmprocs(worker_pid)
                    num_attempted, num_completed, num_correct, num_with_values, exp_finished = take!(status_channel)
                    if !exp_finished
                        num_attempted += 1
                    end
                    put!(status_channel, (num_attempted, num_completed, num_correct, num_with_values, exp_finished))
                    cur_query += 1
                    worker_pid = -1
                    finished = true
                end
            end
        end
        num_attempted, num_completed, num_correct, num_with_values, exp_finished = fetch(status_channel)
        while isready(results_channel)
            push!(results, take!(results_channel))
        end
        results_filename = "Experiments/Results/" * param_to_results_filename(experiment)
        writedlm(results_filename, results, ',')

        total_runtime = sum([parse(Float64,x[4]) for x in results[2:end]])
        total_opt_time = sum([parse(Float64,x[5]) for x in results[2:end]])
        println("Attempted Queries: ", num_attempted)
        println("Completed Queries: ", num_completed)
        println("Queries With Ground Truth: ", num_with_values)
        println("Correct Queries: ", num_correct)
        println("Total Runtime: ", total_runtime)
        println("Total Opt. Time: ", total_opt_time)
        metadata = [("Attempted", "Completed", "HasGroundTruth", "Correct", "TotalRuntime", "TotalOptTime"),
                    (string(num_attempted), string(num_completed), string(num_with_values), string(num_correct), string(total_runtime), string(total_opt_time))]

        metadata_filename = "Experiments/Results/" * param_to_results_filename(experiment, ext=".meta")
        writedlm(metadata_filename, metadata, ',')
    end
end
