

@enum WORKLOAD aids human lubm80 yago yeast yeast_lite hprd hprd_lite wordnet dblp dblp_lite youtube youtube_lite eu2005 patents


const IS_GCARE_DATASET = Dict(aids=>true, human=>true, lubm80=>true, yago=>true,
    yeast=>false, hprd=>false, wordnet=>false, dblp=>false, youtube=>false,
    eu2005=>false, patents=>false)

const IS_SUBGRAPH_WORKLOAD = Dict(aids=>true, human=>true, lubm80=>true, yago=>true,
     yeast=>true, yeast_lite=>true, hprd=>true, hprd_lite=>true, wordnet=>true,
     dblp=>true, dblp_lite=>true, youtube=>true, youtube_lite=>true, eu2005=>true, patents=>true)

struct ExperimentParams
    workload::WORKLOAD
    warm_start::Bool
    faq_optimizer::FAQ_OPTIMIZERS
    stats_type::Type
    use_duckdb::Bool
    use_umbra::Bool
    use_umbra_parallel::Bool
    update_cards::Bool
    simple_cse::Bool
    max_kernel_size::Int
    timeout::Float64
    description::String

    function ExperimentParams(;workload=human, warm_start=false, faq_optimizer=naive, stats_type = DCStats, use_duckdb=false, use_umbra=false, use_umbra_parallel=false, update_cards=true, simple_cse=true, max_kernel_size=8, timeout=60*5, description="")
        return new(workload, warm_start, faq_optimizer, stats_type, use_duckdb, use_umbra, use_umbra_parallel, update_cards, simple_cse, max_kernel_size, timeout, description)
    end
end

function param_to_results_filename(param::ExperimentParams; ext=".csv")
    filename = ""
    filename *= string(param.workload) * "_"
    filename *= string(param.warm_start) * "_"
    filename *= string(param.faq_optimizer) * "_"
    filename *= string(param.stats_type) *  "_"
    filename *= string(param.use_duckdb) *  "_"
    filename *= string(param.use_umbra) *  "_"
    filename *= string(param.use_umbra_parallel) *  "_"
    filename *= string(param.update_cards) * "_"
    filename *= string(param.simple_cse) * "_"
    filename *= string(param.max_kernel_size) * ext
    return filename
end
