
julia --project=. "Experiments/Scripts/matrix_chain_galley.jl"
python "Experiments/Scripts/matrix_chain_pytorch.py" --num-cores 1
python "Experiments/Scripts/matrix_chain_pytorch.py" --num-cores 8