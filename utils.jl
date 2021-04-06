function my_set_submap!(target, addr, source)
    Gen.set_submap!(target, addr, get_submap(source, addr))
end


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




logit(p::Real) = log(p/(1 -p))
ell(p::Real) = logit(p)
expit(ell::Real) = 1 - 1/(1 + exp(ell));


Vec(i::CartesianIndex{2}) = Float64[Tuple(i)...];
Float64(i::CartesianIndex{2}) = Float64[Tuple(i)...];


@dist function labeled_cat(labels, probs)
    index = categorical(probs)
    labels[index]
end;



