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

Base.getindex(x::T, i::Int64) where T <: StructIterator = getfield(x,fieldnames(T)[i])


# ------------------------
#   Gen helper
# ------------------------
function my_set_submap!(target, addr, source)
    Gen.set_submap!(target, addr, get_submap(source, addr))
end


function set_constraint!(target, addr, source::Gen.DynamicChoiceMap)
    Gen.set_submap!(target, addr, get_submap(source, addr))
end;


function set_constraint!(target, addr, tr::Gen.DynamicDSLTrace)     
    source = get_choices(tr)
    set_constraint!(target, addr, source)
end;


changes(bs::Array{T,1}) where T <: Real = Tuple(map(b -> Bool(b) ? UnknownChange() : NoChange(), bs));


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
Gen.has_output_grad(::MultivariateUniform) = false;
Gen.has_argument_grads(::MultivariateUniform) = (false,);


@dist function labeled_cat(labels, probs)
    index = categorical(probs)
    labels[index]
end;


logit(p::Real) = log(p/(1 -p))
ell(p::Real) = logit(p)
expit(ell::Real) = 1 - 1/(1 + exp(ell));
Vec(i::CartesianIndex{2}) = Float64[Tuple(i)...];
Float64(i::CartesianIndex{2}) = Float64[Tuple(i)...];
center(size::Tuple{Int, Int}) = CartesianIndex(Int.(ceil.(size./2))...);
function CartesianIndex(t::Int, size::Tuple{Int64,Int64}, order=:col)
    n = order == :col ? size[1] : size[2]
    i = mod(t-1,n) + 1 
    j = div(t - 1, n) + 1
    return order == :col ? CartesianIndex(i, j) : CartesianIndex(j, i)
end
euclidean(z::Array{Float64,1}, a::Array{Float64,1}) = z .* [cos.(a) sin.(a)]
function norm2(A; dims)
    B = sum(x -> x^2, A; dims=dims)
    B .= sqrt.(B)
end;
s1_diff(a, b) = mod((a - b) + π, 2π) - π;
s1_dist(a, b) = abs(s1_diff(a, b));