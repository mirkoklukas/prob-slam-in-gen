"""
    load_env(i::Union{Int64, Nothing}=nothing; 
             path=p"./data/HouseExpo_json")

Loads the i'th "House Expo" environment from the specified path to 
the corresponding json files. If called with `nothing` it returns 
the number of environments.

The data set can be found at `https://github.com/TeaganLi/HouseExpo`
and is accompanied with an arXiv preprint:

>   Li et al. "HouseExpo: A Large-scale 2D Indoor Layout Dataset for 
>   Learning-based Algorithms on Mobile Robots", arXiv (2019).
    
"""
function load_env(i::Union{Int64, Nothing}=nothing; 
                  path=p"./data/HouseExpo_json")

    files = collect(walkpath(path))    
    if i == nothing return length(files) end
    
    env = Dict()
    open(files[i], "r") do f
        env = JSON.parse(read(f, String))
    end

    return Env(env) 
end



"""
    get_verts(env)

Returns an array of shape (n,2) containing the vertices
of the env.
"""
get_verts(env) = transpose(hcat(env["verts"]...))


"""
    get_segs(env)

Returns an array of shape (n, 4) encoding
a closed polygon outlining the wall segments of the env.
"""
function get_segs(env)
    v = transpose(hcat(env["verts"]..., env["verts"][1]))
    s = hcat(v[1:end-1,:], v[2:end,:])
    return s
end


function get_bounds(env)
    b = [env["bbox"]["min"][1] env["bbox"]["max"][1]; 
         env["bbox"]["min"][2] env["bbox"]["max"][2]]
    return b
end

"""
    Env

Struct containing the vertices, wall segments, and
the bounding box of an environment.
"""
struct Env
    verts::Array{Float64, 2}
    segs::Array{Float64, 2}
    bounds::Array{Float64, 2}
    Env(env::Dict) = new(get_verts(env), get_segs(env), get_bounds(env))
end


