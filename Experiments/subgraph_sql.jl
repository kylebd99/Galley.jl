using LibPQ, Tables
using Galley
using Finch
using IterTools: imap
using DataStructures: counter, inc!
using CSV

tensor_id_to_table_name(tensor_id, table_name_counters) = "tensor_$(tensor_id)_$(table_name_counters[tensor_id])"

function create_table_umbra(conn, idxs, table_name)
    execute(conn, "DROP TABLE IF EXISTS $table_name")
    create_str = "CREATE TABLE $table_name ("
    prefix = ""
    for idx in idxs
        create_str *= "$prefix $(idx) INT"
        prefix = ", "
    end
    create_str *= "$prefix v INT)"
    create_str = replace(create_str, "#"=>"")
    execute(conn, create_str)
end

function tensor_to_table(t::Tensor, idxs)
    return  zip(ffindnz(t)...)
end

function fill_table_umbra(conn, tensor, idxs, table_name)
    create_table_umbra(conn, idxs, table_name)
    tensor_data = tensor_to_table(tensor, idxs)
    rows = [NamedTuple{Tuple([idxs..., :v])}(idx_and_vals) for idx_and_vals in tensor_data]
    csv_filename = "/tmp/$table_name.csv"
    CSV.write(csv_filename, rows)
    run(`psql -p 5432 -h 127.0.0.1 -U postgres password=postgres -c "\copy $table_name FROM '$csv_filename' CSV HEADER NULL 'na'"`)
end

function get_sql_query(table_names, table_idxs; explain_query = false)
    table_aliases = ["A_$i" for i in eachindex(table_names)]
    sql_query = explain_query ? "EXPLAIN SELECT COUNT(*)" : "SELECT COUNT(*)" 

#=
    sql_query = explain_query ? "EXPLAIN SELECT SUM(" : "SELECT SUM(" 
    prefix = ""
    for name in table_aliases
        sql_query *= "$prefix $name.v"
        prefix = " *"
    end
     sql_query *= ")"
 =#
    idx_to_root_alias = Dict()
    for i in eachindex(table_names)
        for idx in table_idxs[i]
            if !haskey(idx_to_root_alias, idx)
                idx_to_root_alias[idx] = table_aliases[i]
            end
        end
    end

    table_sub_query = if length(table_idxs[1]) == 1
        "(SELECT i as $(table_idxs[1][1]), v
        FROM $(table_names[1]))"
    else
        "(SELECT i as $(table_idxs[1][1]), j as $(table_idxs[1][2]), v
        FROM $(table_names[1]))"
    end

    sql_query *= " \nFROM $(table_sub_query) as $(table_aliases[1])\n"
    for i in 2:length(table_names)
        prefix = ""
        table = table_names[i]
        alias = table_aliases[i]
        table_sub_query = if length(table_idxs[i]) == 1
            "(SELECT i as $(table_idxs[i][1]), v
            FROM $table)"
        else
            "(SELECT i as $(table_idxs[i][1]), j as $(table_idxs[i][2]), v
            FROM $table)"
        end
        sql_query *= "INNER JOIN $table_sub_query as $(alias) ON ("
        has_clause = false
        for idx in table_idxs[i]
            if alias != idx_to_root_alias[idx]
                sql_query *= "$prefix $alias.$idx = $(idx_to_root_alias[idx]).$idx"
                prefix = " AND"
                has_clause = true
            end
        end
        if !has_clause
            sql_query *= "true"
        end
        sql_query *= ")\n"
    end
    sql_query *= ";"
    return sql_query    
end

# The query needs to take the form of:
# Query(:out, Materialize(Aggregate(+, MapJoin(*, Input(TensorId, idxs...)...))))
# i.e. no tree of mapjoin operators
function execute_galley_query_umbra(query::PlanNode, use_parallel=false)
    conn = LibPQ.Connection("host=127.0.0.1 port=5432 user=postgres password=postgres")
    if use_parallel
        execute(conn, "set debug.parallel=24")
    else
        execute(conn, "set debug.parallel=1")
    end
    aggregate = query.expr.expr
    mapjoin = aggregate.arg
    inputs = mapjoin.args
    table_names = []
    table_name_counters = counter(String)
    table_idxs = []
    insert_start = time()
    for input in inputs
        @assert input.kind == Input
        idxs = [idx.name for idx in input.idxs]
        inc!(table_name_counters, input.id)
        table_name = input.id
        push!(table_names, table_name)
        push!(table_idxs, idxs)
    end
    explain_query = get_sql_query(table_names, table_idxs; explain_query=true)
    println(explain_query)
    compile_start =  time() 
    explain_result = execute(conn, get_sql_query(table_names, table_idxs; explain_query=true))
    compile_time = time() - compile_start
    opt_start = time() 
    explain_result = execute(conn, get_sql_query(table_names, table_idxs; explain_query=true))
    opt_time = time() - opt_start
    println(columntable(explain_result).plan[1])
    sql_query = get_sql_query(table_names, table_idxs)

    result = columntable(execute(conn, sql_query))
    execute_start = time() 
    result = columntable(execute(conn, sql_query))
    execute_time = time()-execute_start - opt_time


#    for table_name in table_names
#        execute(conn, "DROP TABLE $table_name")
#    end
    println("Compile Time: $compile_time")
    println("Opt Time: $opt_time")
    println("Execute Time: $execute_time")
    return (result = result[1], execute_time = execute_time, opt_time = opt_time, compile_time=compile_time-opt_time)
end
