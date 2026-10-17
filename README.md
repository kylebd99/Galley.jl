# Galley.jl

Galley is a system for declarative sparse tensor algebra. It combines techniques from database query optimization with the flexible formats and execution of sparse tensor compilers, specifically the [Finch compiler](https://github.com/willow-ahrens/Finch.jl). Details about the theory and system design can be found in our paper [here](https://arxiv.org/abs/2408.14706). Currently, Galley vendors Finch.jl, but we are working to make Finch.jl a proper dependency.

## Galley: An Optimizer for Finch.jl

Using Galley with Finch's array API just requires setting the `ctx` parameter to `galley_scheduler()` in the `compute` function!

```
using Finch
using Galley

A = lazy(Tensor(Dense(SparseList(Element(0.0))), fsprand(100, 100, .1)))
B = lazy(Tensor(Dense(SparseList(Element(0.0))), fsprand(100, 100, .1)))
C = compute(A * B, ctx = galley_scheduler())
```

Alternatively, you can specify that Galley should be used at the top of the file:

```
using Finch
using Galley

Finch.set_scheduler!(galley_scheduler())
A = lazy(Tensor(Dense(SparseList(Element(0.0))), fsprand(100, 100, .1)))
B = lazy(Tensor(Dense(SparseList(Element(0.0))), fsprand(100, 100, .1)))
C = compute(A * B)
```

This is the preferred usage pattern for Galley.

## Galley: Direct Usage

Instead of going through the array API, you can specify your computation directly with Galley. This uses the grammar defined in `src/plan.jl`. For example, we can specify the same matrix multiplication as above using:

```
using Finch 
using Galley

A = Tensor(Dense(SparseList(Element(0.0))), fsprand(100, 100, .1))
B = Tensor(Dense(SparseList(Element(0.0))), fsprand(100, 100, .1))
C = galley(Mat(:i, :k, Agg(+, :j, MapJoin(*, Input(A, :i, :j), Input(B, :j, :k))))).value[1]
C == compute(lazy(A) * B, ctx=galley_scheduler()) # True
```
