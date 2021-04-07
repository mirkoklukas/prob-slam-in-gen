

abstract type Cell end


"""
    MapCell(x,v,i)

A single grid cell in an occupancy map.
"""
mutable struct MapCell <: Cell
    x::Vector{Float64}
    v::Float64
    i::CartesianIndex
end
# Rem: Could add resolution to each cell...
# ---

"""
    Map(cells, res)

An occupancy map containing cells of a fixed resolution.
(This could be made variable, and put on the individual cells.)
"""
mutable struct Map
    cells::Array{MapCell}
    res::Float64
end;

Base.size(M::Map) = size(M.cells)
Base.size(M::Map, d::Int) = size(M.cells, d)
Base.iterate(M::Map) = iterate(M.cells)
Base.iterate(M::Map, state) = iterate(M.cells, state)
Base.getindex(M::Map, i) = M.cells[i]
Base.length(M::Map) = length(M.cells)
occupied(M::Map) = filter(c -> c.v == 1., M.cells)
walkable(M::Map) = filter(c -> c.v == 0., M.cells)
pos(c::MapCell) = c.x
val(c::MapCell) = c.v
pos(M::Map) = hcat(pos.(M.cells)...)
center(M::Map) = mean(pos.(M.cells))
function CartesianIndex(t::Int, M::Map)
    n = size(M.cells,1)
    i = mod(t-1,n) + 1 
    j = div(t - 1, n) + 1
    CartesianIndex(i, j)
end

center(size::Tuple{Int, Int}) = CartesianIndex(Int.(ceil.(size./2))...);
euclidean(z::Array{Float64,1}, a::Array{Float64,1}) = z .* [cos.(a) sin.(a)]
function norm2(A; dims)
    B = sum(x -> x^2, A; dims=dims)
    B .= sqrt.(B)
end;

s1_diff(a, b) = mod((a - b) + π, 2π) - π;
s1_dist(a, b) = abs(s1_diff(a, b));

function draw_map!(M::Map; ax=plt.gca(),m="o", s=nothing, alpha=.3, c="black", zorder=0)    
    ax.set_aspect(1)    
    occ = occupied(M)
    if length(occ) > 0
        x = hcat(pos.(occ)...)
        ax.scatter(x[1,:],x[2,:], s=s, c=c,marker=m, alpha=alpha, zorder=zorder)
    end
end;

function draw_ell_map!(M::Map; ax=plt.gca(), m="o", s=nothing, alpha=1., cmap="binary", zorder=0)    
    ax.set_aspect(1)    
    x = hcat(pos.(M.cells)...)
    v = expit.(val.(M))
    ax.scatter(x[1,:],x[2,:], s=s, c=v, marker=m, alpha=alpha, cmap=cmap, vmin=0, vmax=1, zorder=zorder)
end;


function draw_map_im!(im::Array{T,2}; ax=plt.gca(), cmap="binary", vmin=-1, vmax=1) where T<: Real
    im = transpose(im)
    im = im[end:-1:1,:]
    ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
end;


function draw_map_circles!(M::Map; ax=plt.gca(), alpha=.2, cmap="binary", zorder=0)     
    cm = plt.cm.get_cmap(cmap)
    ax.set_aspect(1);
    for c in occupied(M)
        circle1 = plt.Circle(c.x, M.res/2, color=cm(c.v), alpha=alpha, zorder=zorder)
        ax.add_patch(circle1)
    end
end;

function init_map(map_size::Tuple{Int,Int}, 
                  res::Float64,
                  v0::Float64)
    
    i0 = center(map_size)
    cs = Array{MapCell,2}(undef, map_size...)
    for i in CartesianIndices(map_size)    
        cs[i] = MapCell(res.*Vec(i - i0), v0, i)
    end
    return Map(cs, res)
end;

"""
    map_prior(map_size::Tuple{Int,Int}, 
              res::Float64, 
              pr::Float64) -> M::Map

Returns map of occupancy grid cells....

Choices:
```
    | 
    +-- (:m, CartesianIndex): Bool
```
"""
@gen function map_prior(map_size::Tuple{Int,Int}, 
                        res::Float64, 
                        pr::Float64)
    
    i0 = center(map_size)
    cs = Array{MapCell,2}(undef, map_size...)
    for i in CartesianIndices(map_size)    
        v = @trace(bernoulli(pr), (:m, i))
        cs[i] = MapCell(res.*Vec(i - i0), v, i)
    end

    return Map(cs, res)
end;


"""
    Pose

Encoding the pose of a 2d agent as 
a 2d-position `x` and its head direction `hd`.
"""
struct Pose <: StructIterator
    x::Array{Float64}
    hd::Float64
end;

"""
    pose_prior(M::Map)

Samples a pose where the position is uniformly
chosen from the un-occupied cell-centers.

Choices:
```
    | 
    +-- (:hd, Float64)
    |
    +-- (:x,  Array{Float64})
```
"""
@gen function pose_prior(M::Map)
    free = walkable(M)
    len = length(free)
    
    hd ~ uniform(0,2π)
    x  ~ labeled_cat([pos.(free)...], ones(len)/len)
    return Pose(x,hd)
end;
# Rem: Note that pr(hd = -pi) = 0. We need to define a better distr over the unitcircle!
# ---  This was the source of a bug which took quite a bit to unravel...

function draw_pose!(pose; ax=plt.gca(), c="black", m="o", l=1., zorder=1)
    x, hd = pose
    nose = l.*[cos(hd) sin(hd)]
    ax.scatter(x..., c="w", marker="o", edgecolor=c, zorder=zorder+1)
    ax.plot([x[1];x[1]+nose[1]], [x[2];x[2]+nose[2]], c=c, zorder=zorder)
end;


"""
    angles(fov::Int64=180, n=100)

Range of angles from `-fov` degrees to `+fov` degrees,
however the entries are in radians not degrees.
"""
function angles(fov::Int64, n::Int)
    if fov > 180 || fov <= 0
       throw(DomainError(fov, "fov must satisfy `0 < fov <= 180`")) 
    end
    
    θ  = fov/180*π
    θs = collect(range(-θ, θ, length=n))
    return θs
end;

"""
    (Y, Z, D) = project(X, pose::Pose, a)

Project the columns of `X` onto the bouquet of lines 
based at `pose.x` and spanned by `a`.

The array `Y[i,:,j]` contains the projection of `X[:,j]`
onto the line based at `x` and spanned by `e^{i*(a[i] + hd)}`
where `(x, hd) = pose`...
"""
function project(X::Array{Float64,2}, pose::Pose, a::Array{Float64})
    x, hd = pose
    n = length(a)
    a = a .+ hd
    L = [cos.(a) sin.(a)] # size = (n,2)
    X = X .- x            # size = (2,c)
    Z = L * X             # size = (n,c)
    Y = reshape(Z, n, 1, :) .* L;   # size = (n,2,c)
    V = Y .- reshape(X,1,2,:)
    D = norm2(V, dims=2)[:,1,:]
    return Y, Z, D
end;

"""
    Measurement(z,a)

Sensor measurement containing depth values `z` and
the angles `a` of the simulated laser beams.
"""
struct Measurement <: StructIterator
    z::Vector{Float64} # range measurements
    a::Vector{Float64} # angles (pre-noise)
end;

"""
    occ_lidar(occ::Array{MapCell}, 
              pose::Pose, 
              eps::Float64, 
              fov::Int, 
              n::Int, 
              max_val::Float64)

Range sensor model for an occupancy map. Returns a 
range measuremnt at a pose form an array of occupied
map cells.
"""
function occ_lidar(occ, pose, eps, fov, n, max_val)
    a = angles(fov, n)     
    z = zeros(n)
    
    if length(occ) ==0
        z .= max_val
        return Measurement(z, a)
    end
    

    X = hcat(pos.(occ)...)
    Y,Z,D = project(X, pose, a)
    

    for i=1:n
        
        valid = (Z[i,:] .> 0.0) .* (D[i,:] .<= eps)
        
        if sum(valid) > 0
            z[i] = min([max_val; norm2(Y[i,:,valid], dims=1)]...)
        else
            z[i] = max_val
        end
    end
    return Measurement(z, a)
end;

"""
    occ_lidar_2(occ::Array{MapCell}, 
              pose::Pose, 
              eps::Float64, 
              fov::Int, 
              n::Int, 
              max_val::Float64)

Range sensor model for an occupancy map. Returns a 
range measuremnt at a pose form an array of occupied
map cells.
"""
function occ_lidar_2(occ, pose, eps, fov, n, max_val)
    a = angles(fov, n) .+ pose.hd 
    z = max_val .* ones(n)

    if length(occ) ==0
        return Measurement(z, a)
    end
    
    
    for c in occ
        x = c.x - pose.x
        b = atan(x[2], x[1])
        k = argmin(s1_dist.(b, a))
        z[k] = min(z[k], norm(x))
    end

    return Measurement(z, a .- pose.hd )
end;

function draw_sense!(z,a, pose; ax=plt.gca(), cmap=nothing, cs=nothing, zorder=1)
    y = euclidean(z, a .+ pose.hd) .+ pose.x'
    x = pose.x
        
    if cmap == nothing || cs ==nothing
        for i=1:n
            ax.plot([x[1], y[i,1]],[x[2], y[i,2]], c="C1", alpha=0.2, zorder=zorder);
        end
        ax.scatter(y[:,1], y[:,2], c="C1", alpha=1., s=1, zorder=zorder);
    else
        cm = plt.cm.get_cmap(cmap)
        for i=1:n
            ax.plot([x[1], y[i,1]],[x[2], y[i,2]], c=cm(cs[i]), alpha=0.2, zorder=zorder);
            ax.scatter(y[i,1], y[i,2], c=cm(cs[i]), alpha=1., s=1, zorder=zorder);
        end

            
    end
end;



"""
    Measurement(z, a) = sensor_model(M::Map, pose::Pose, fov, n, sig, max_val)

Range sensor model for an occupancy map.

Choices:
```
    |
    +-- :z : Array{Float64, 1}
```
"""
@gen function sensor_model(M::Map, pose::Pose, fov, n, sig, max_val)
    
    eps = M.res/2
    occ = occupied(M)

    mu_z, a = occ_lidar(occ, pose, eps, fov, n, max_val)
    
    sig_z = sig.*Diagonal(ones(n))
    z  = @trace(mvnormal(mu_z, sig_z), :z)
    
    return Measurement(z, a)
end;


function inverse_lidar(z, a, pose, M)
    im = zeros(size(M)...)
    Y,Z,D = project(pos(M), pose, a);
    eps = M.res/2
    for i=1:length(a)
        z_cond = Z[i,:] .> 0
        eps_cond = (D[i,:] .< eps)
        cond = z_cond .* eps_cond
        occ  = cond .* (abs.(Z[i,:] .- z[i]) .< eps) 
        free = cond .* (Z[i,:] .- z[i] .+ eps .< 0)
        im[occ]  .+= 1
        im[free] .-= 1
    end
    return im
end


function inverse_lidar_2(z, a, pose, M)
    im = zeros(size(M)...)    
    Y,Z,D = project(pos(M), pose, a);
    eps = (M.res/2)

    D[Z.<0] .= Inf
    for j=1:length(M)
        i = argmin(D[:,j])
        if Z[i,j] < z[i] - eps
            im[j] -= 1
        elseif abs(Z[i,j] -  z[i]) <= eps
            im[j] += 1
        end
    end
    return im

end


function inverse_lidar_3(z, a, pose, M)
    Obs = init_map(size(M), M.res, ell(0.5))
    eps = M.res/2

    for c in Obs.cells
        x = c.x - pose.x
        b = atan(x[2], x[1]) .- pose.hd
        k = argmin(s1_dist.(b, a))

        if norm(x) < z[k] - eps
            c.v = ell(0.2)
        elseif abs(norm(x) - z[k]) < eps
            c.v = ell(0.8)
        end

    end
    im = expit.(val.(Obs.cells))
    return im
end


"""
    update!(M::Map, pose::Pose, sense::Measurement; 
            prior=ell(0.5), free=ell(0.2), occ=ell(0.8))

Inverse sensor model and map update following the recipe in  
Table 9.1 and 9.2 in [Thrun et al., 2006].
"""
function update!(M::Map, pose::Pose, sense::Measurement; prior=ell(0.5), free=ell(0.2), occ=ell(0.8))
    z,a = sense
    eps = M.res/2
    for c in M.cells
        x = c.x - pose.x
        b = atan(x[2], x[1]) .- pose.hd
        k = argmin(s1_dist.(b, a))
        
        l = prior
        if norm(x) < z[k] - eps
            l = free
        elseif abs(norm(x) - z[k]) < eps
            l = occ
        end
        # Rem: Bayesian update from Table 9.1 in [Thrun et al., 2006].
        c.v = c.v + l - prior
    end
end


"""
    occ_map::Map = map_model(ell_map::Map)

Samples an occupancy map based on a map that contains cells whose
values are logits of occupancy probabilities.
"""
@gen function ell_map_model(ell_map::Map)
    map_size = size(ell_map)
    res = ell_map.res
    i0 = center(map_size)
    cs = Array{MapCell,2}(undef, map_size...)
    for i in CartesianIndices(map_size)    
        pr = expit(ell_map[i].v)
        v = @trace(bernoulli(pr), (:m, i))
        cs[i] = MapCell(ell_map[i].x, v, i)
    end

    return Map(cs, res)
end;


