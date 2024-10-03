# This file is a part of Julia. License is MIT: https://julialang.org/license
using Base: union!, union, setdiff!, setdiff, intersect!, intersect, eltype, empty
using Base: emptymutable, copy, copymutable,push!, delete!, empty!, isempty,in,issubset
using Base: ⊊,==,iterate, length, hash
Bits = UInt128
struct SmallBitSet <: AbstractSet{Int}
    bits::Bits
    SmallBitSet() = new(0)
    SmallBitSet(b::Bits) = new(b)
end

"""
    SmallBitSet([itr])

Construct a sorted set of `Int`s generated by the given iterable object, or an
empty set. Implemented as a static bit string, and therefore designed for integers between
1 and 128.
"""
function SmallBitSet(ints::Vector{Int})
    s = SmallBitSet()
    for i in ints
        s = _setint(s, i, true)
    end
    return s
end

# Special implementation for BitSet, which lacks a fast `length` method.
function Base.union(s::SmallBitSet, itr)
    for i in itr
        s = _setint(s, i, true)
    end
    return s
end

Base.eltype(::Type{SmallBitSet}) = Int

Base.empty(s::SmallBitSet, ::Type{Int}=Int) = SmallBitSet()

function Base.copy(src::SmallBitSet)
    dest.bits = src.bits
    dest
end

@inline _mod128(l) = l & 127
function _bits_getindex(b::Bits, n::Int)
    @inbounds r = (b & (one(UInt128) << _mod128(n))) != 0
    r
end

# An internal function for setting the inclusion bit for a given integer
@inline function _setint(s::SmallBitSet, idx::Int, b::Bool)
    u = UInt128(1) << idx
    SmallBitSet(ifelse(b, s.bits | u, s.bits & ~u))
end

@noinline _throw_bitset_bounds_err() =
    throw(ArgumentError("elements of SmallBitSet must be between 0 and 128"))

@inline _is_convertible_Int(n) = 0 < n <= 128

@inline _check_bitset_bounds(n) =
    _is_convertible_Int(n) ? Int(n) : _throw_bitset_bounds_err()

@inline _check_bitset_bounds(n::Int) = n

@noinline _throw_keyerror(n) = throw(KeyError(n))

Base.isempty(s::SmallBitSet) = s.bits == 0

# Mathematical set functions: union, intersect, setdiff
function Base.union(s::SmallBitSet, s2::SmallBitSet)
    SmallBitSet(s.bits | s2.bits)
end

# Mathematical set functions: union, intersect, setdiff
function Base.union(s::SmallBitSet, sets...)
    b::Bits = s.bits
    for s1 in sets
        b |= s1.bits
    end
    SmallBitSet(b)
end

function Base.intersect(s1::SmallBitSet, s2::SmallBitSet)
    SmallBitSet(s1.bits & s2.bits)
end

function Base.setdiff(s1::SmallBitSet, s2::SmallBitSet)
    SmallBitSet(s1.bits & ~s2.bits)
end

@inline Base.in(n::Integer, s::SmallBitSet) = _is_convertible_Int(n) ? _bits_getindex(s.bits, Int(n)) : false

function Base.iterate(s::SmallBitSet, idx = 0)
    word = 0
    while word == 0
        idx == 128 && return nothing
        idx += 1
        word = s.bits & (one(UInt128) << idx)
    end
    idx, idx
end

function Base.length(s::SmallBitSet)
    count_ones(s.bits)
end

@noinline _throw_bitset_notempty_error() =
    throw(ArgumentError("collection must be non-empty"))

function Base.:(==)(s1::SmallBitSet, s2::SmallBitSet)
    return s1.bits == s2.bits
end

Base.issubset(a::SmallBitSet, b::SmallBitSet) = a == intersect(a,b)
Base.:(⊊)(a::SmallBitSet, b::SmallBitSet) = a <= b && a != b

function Base.hash(s::SmallBitSet)
    return hash(s.bits)
end

function Base.hash(s::SmallBitSet, h::UInt128)
    return hash(s.bits, h)
end

function Base.hash(s::SmallBitSet, h::UInt64)
    return hash(s.bits, h)
end
