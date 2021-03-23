"""
    StructIterator

A quick helper to enable destructuring assignments for my types.
"""
abstract type StructIterator end

function Base.iterate(x::T) where T <: StructIterator  
    n = fieldnames(T)
    getfield(x, n[1]), 2
end
    
function Base.iterate(x::T, state) where T <: StructIterator 
    n = fieldnames(T)
    if state <= length(n)
        return getfield(x, n[state]), state + 1
    else
        return nothing
    end
end


"""
    MultivariateUniform

Mulitivariate uniform distributions (surprise!).
"""
struct MultivariateUniform <: Gen.Distribution{Vector{Float64}} end
const mvuniform = MultivariateUniform()

function Gen.logpdf(::MultivariateUniform, x::AbstractArray{Float64,1}, b::AbstractArray{Float64,2})
    dist = Product(Uniform.(b[:,1], b[:,2]))
    Distributions.logpdf(dist, x)
end

function Gen.random(::MultivariateUniform, b::AbstractArray{Float64,2})
    dist = Product(Uniform.(b[:,1], b[:,2]))
    rand(dist)
end

(::MultivariateUniform)(b) = Gen.random(MultivariateUniform(), b)

has_output_grad(::MultivariateUniform) = false;
has_argument_grads(::MultivariateUniform) = (false,);
