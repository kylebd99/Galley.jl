cd tpch-dbgen
./dbgen -s .25
mv *.tbl ../Experiments/Data/TPCH 
./dbgen -s 5 
mv *.tbl ../Experiments/Data/TPCH_5 
cd ..

poetry run python "Experiments/Scripts/tpch_inference_polars.py" --parallel true

poetry run python "Experiments/Scripts/tpch_inference_polars.py" --parallel false

poetry run python "Experiments/Scripts/tpch_inference_pandas.py"

julia --project=. "Experiments/Scripts/tpch_inference.jl"