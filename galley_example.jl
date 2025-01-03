using Galley
using Finch
using Finch: countstored
using BenchmarkTools 

for N in [100, 200, 300]
    println("Generating Data")
    A = Tensor(Dense(SparseList(Element(zero(UInt64)))), rand(UInt64, N, N) .% 128)
    B = Tensor(Dense(SparseList(Element(zero(UInt64)))), rand(UInt64, N, N).% 128)
    C = Tensor(Dense(SparseList(Element(zero(UInt64)))), fsprand(UInt64, N, N, .01).% 128)
    println("Counstored(A): $(countstored(A))")
    println("Counstored(B): $(countstored(B))")
    println("Counstored(C): $(countstored(C))")
    A = lazy(A)
    B = lazy(B)
    C = lazy(C)
    
    println("Galley: A * B * C")
    empty!(Finch.codes)
    @btime begin 
        compute($A * $B * $C, ctx=galley_scheduler())
    end
    compute(A * B * C, ctx=galley_scheduler(verbose=true))

    println("Galley: C * B * A")
    empty!(Finch.codes)
    @btime begin 
        compute($C * $B * $A, ctx=galley_scheduler())
    end
    compute(C * B * A, ctx=galley_scheduler(verbose=true))

    println("Galley: sum(C * B * A)")
    empty!(Finch.codes)
    @btime begin 
        compute(sum($C * $B * $A), ctx=galley_scheduler())
    end
    compute(sum(C * B * A), ctx=galley_scheduler(verbose=true))

    println("Finch: A * B * C")
    empty!(Finch.codes)
    @btime begin 
        compute($A * $B * $C, ctx=Finch.default_scheduler())
    end

    println("Finch: C * B * A")
    empty!(Finch.codes)
    @btime begin 
        compute($C * $B * $A, ctx=Finch.default_scheduler())
    end

    println("Finch: sum(C * B * A)")
    empty!(Finch.codes)
    @btime begin 
        compute(sum($C * $B * $A), ctx=Finch.default_scheduler())
    end

    println((empty!(Finch.codes); compute(A * B * C, ctx=galley_scheduler())) == (empty!(Finch.codes); compute(A * B * C, ctx=Finch.default_scheduler())))
    println((empty!(Finch.codes); compute(C * B * A, ctx=galley_scheduler())) == (empty!(Finch.codes); compute(C * B * A, ctx=Finch.default_scheduler())))
    println((empty!(Finch.codes); compute(sum(C * B * A), ctx=galley_scheduler())) == (empty!(Finch.codes); compute(sum(C * B * A), ctx=Finch.default_scheduler())))
end



A = Tensor(Dense(Element(0), 10))
prgm = @finch_program_instance begin 
    A .= 0
    for i = _
        A[i] += 1
    end
end
eval(begin
   @finch $(finch_unparse_program(nothing, prgm))
end)
A1 = A
A = Tensor(Dense(Element(0), 10))
prgm = @finch begin 
    A .= 0
    for i = _
        A[i] += 1
    end
end
A == A1