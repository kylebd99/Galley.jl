cd tpch-dbgen
./dbgen -s .01
mv *.tbl ../Experiments/Data/TPCH 
./dbgen -s .01
mv *.tbl ../Experiments/Data/TPCH_5 
cd ..

mkdir -p Experiments/Results

poetry run python "Experiments/Scripts/tpch_inference_polars.py" --parallel False

poetry run python "Experiments/Scripts/tpch_inference_polars.py" --parallel True

poetry run python "Experiments/Scripts/tpch_inference_pandas.py"

julia --project=. "Experiments/Scripts/tpch_inference.jl"