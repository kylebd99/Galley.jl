# Overview
Galley is a system for declarative sparse tensor algebra. It combines techniques from database query optimization with the flexible formats and execution of sparse tensor compilers, specifically the [Finch compiler](https://github.com/willow-ahrens/Finch.jl). Details about the theory and system design can be found in our paper [here](https://dl.acm.org/doi/pdf/10.1145/3725301). 

Note: Finch needs to be imported because Galley relies on it for both data loading and for compiling sparse tensor functions.

## Galley: Direct Usage
You can specify your computation directly with Galley. This uses the grammar defined in `src/plan.jl`. For example, we can specify the same matrix multiplication as above using:

```
using Finch 
using Galley

A = Tensor(Dense(SparseList(Element(0.0))), fsprand(100, 100, .1))
B = Tensor(Dense(SparseList(Element(0.0))), fsprand(100, 100, .1))
C = galley(Mat(:i, :k, Agg(+, :j, MapJoin(*, Input(A, :i, :j), Input(B, :j, :k))))).value[1]
C == compute(lazy(A) * B, ctx=galley_scheduler()) # True
```
## Reproducibility

To reproduce the experiments from this paper, you will need to install julia and poetry (for managing the python dependencies). To install python dependencies run:
```
poetry install
```
To run the join inference experiments, run:
```
bash run_inference_experiments.sh
```
To run the matrix experiments, run:
```
bash run_matrix_experiments.sh
```
To run the subgraph experiments, run (note: We restrict it to the human and aids benchmarks to reduce the runtime. To run the other benchmarks, comment out line 4 of Experiments/Benchmark/Large/subgraph_benchmark.jl):
```
bash run_subgraph_experiments.sh
```
To run the bfs experiments, run:
```
bash run_bfs_experiments.sh
```

These experiments all produce their graph in Experiments/Figures.

Note: Some of these experiments require significant memory to run. The machine that it was originally evaluated on had 256 GB of memory. This is necessary for evaluating the pandas/numpy and polars/pytorch comparisons in the join inference experiments.

