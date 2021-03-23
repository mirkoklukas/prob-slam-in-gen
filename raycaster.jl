"""
    X = line_intersection(s::Array{Float64,1}, Q ::Array{Float64,2})

Computes the intersections of a line given by the segment `s=[x;y]`
(which is the concatenation of a base vector x and another point y 
on the line) with every row-segments in the 2d Array `Q`.

Each row in `X` encodes the intersection with the corresponind row 
in `Q`.
"""
function line_intersection(s::Array{Float64,1}, Q ::Array{Float64,2})

    p , q  = s[1:2]  , s[3:4]
    p′, q′ = Q[:,1:2], Q[:,3:4]

    x1, y1 = p
    x2, y2 = q

    x3, y3 = p′[:,1], p′[:,2]
    x4, y4 = q′[:,1], q′[:,2]


    nx = (x1.*y2 .- y1.*x2).*(x3 .- x4) .- (x1 .- x2).*(x3.*y4 .- y3.*x4)
    ny = (x1.*y2 .- y1.*x2).*(y3 .- y4) .- (y1 .- y2).*(x3.*y4 .- y3.*x4)

    d = (x1 .- x2).*(y3 .- y4) .- (y1.-y2).*(x3 .- x4)  # .. math:: d = det (dp, dq) 
    X = [(nx./d) (ny./d)]
    return X
end;


import LinearAlgebra: dot
"""
    (X, C, S, T) = ray_coll(p::Array{Float64,1}, Q::Array{Float64,2})

Computes the intersections of a line (light ray) given by the segment `p=[x;y]`
(which is the concatenation of a base vector x and another point y 
on the line) with every row-segments in the 2d Array `Q`.

`x=X[i,:]` is the intersection of the ray and the line spanned by `q=Q[i,:]`.
`C[i]` is a boolean indicating whether the ray hit the segment `q`.
`s=S[i]` and `t=T[i]` are scalars such that `s*dp = t*q = x`
"""
function ray_coll(p::Array{Float64,1}, Q::Array{Float64,2})

    X = line_intersection(p, Q)

    dp = p[3:4]   .- p[1:2]
    dQ = Q[:,3:4] .- Q[:,1:2]
    
    V = X[:,1:2] .- reshape(p[1:2], 1, 2)

    S = (V*dp)/dot(dp, dp)
        
    V = X - Q[:,1:2]

    T = sum(V.*dQ, dims=2)./sum(dQ.^2, dims=2)
    T = T[:,1]

    C = (0 .<= T).*(T.<=1).*(S.>=0)

    return X, C, S, T
end;


"""
    angles(fov::Int64=180, n=100)

Range of angles from `-fov` degrees to `+fov` degrees,
however the entries are in radians not degrees.
"""
function angles(fov::Int64=180, n=100)
    θ  = fov/180*π
    θs = collect(range(-θ, θ, length=n))
    return θs
end


function lightcone(θs, r=1.0)    
    n = size(θs,1)
    ys = r .* hcat(cos.(θs), sin.(θs))
    return hcat(zeros(n,2), ys)
end

"""
    cast(rays, x, env::Env; max_val=Inf)

Cast a bunch of rays from `x` in `env`.
Returns a vector of depths of their intersection
with the environment segments.
"""
function cast(rays, x, env::Env; max_val=Inf)
    n = size(rays, 1)
    z = zeros(n)
    for i in 1:n
        r = rays[i,:] .+ [x; x]
        X, C, S, T = ray_coll(r, env.segs)
        if sum(C) > 0
            z[i] = min(S[C]..., max_val)
        else
            z[i] = max_val
        end
    end
    return z
end


"""
Compute the intersections points from rays 
and their depths.
"""
get_hits(rays, z) = z .* rays[:,3:4]


function draw_rays!(rays, z, x; every=10,c="C1", alpha=0.1, ax=plt.gca())
    hits = z .* rays[:,3:4] .+ x'
    
    n = size(hits,1)
    sub = collect(1:every:n)
    r = [ones(n,2).*x' hits][sub,:]
    ax.plot(r[:,[1, 3]]', r[:,[2, 4]]', c=c, alpha=alpha, linewidth=1, zorder=0); 
end;


"""
    is_inside(x, segs)

Checks whether `x` lies within the polygon described 
by the segments `segs`.
"""
function is_inside(x, segs)
    X, C, S, T = ray_coll([x;x+[rand(),rand()]], segs)
    return sum(C)%2!=0
end;