"""
    Pose

Encoding the pose of a 2d agent as 
a 2d-position `x` and its head direction `hd`.
"""
struct Pose <: StructIterator
    x::Array{Float64}
    hd::Float64
end

"""
    pose_prior(env)

Samples a pose in the environment. Position
is sampled within the environment bounds and can
potentially land on the outside lawn.
"""
@gen function pose_prior(env::Env)
    x  ~ mvuniform(env.bounds)
    hd ~ uniform(0,2π)
    return Pose(x, hd)
end;


"""
    sensor_model(pose, 
                 env, 
                 n, 
                 fov, 
                 angular_noise, 
                 depth_noise,
                 max_depth)

Samples a vector of a pseudo lidar measurement
given a pose in an environment.
"""
@gen function sensor_model(pose::Pose, env::Env, n, fov, angular_noise, depth_noise, max_depth)
    x, rot = pose
    Iₙ = Diagonal(ones(n))
    
    σₐ = (angular_noise/180)*π
    Σₐ = σₐ .* Iₙ 
    
    a′ = angles(fov, n) .+ rot
    a  = @trace(mvnormal(a′, Σₐ), :ray_directions)

    
    z′ = cast(lightcone(a), x, env; max_val=max_depth)
    σᵣ = depth_noise.*z′ .+ 0.000001
    Σᵣ = σᵣ .* Iₙ
    z  = @trace(mvnormal(z′, Σᵣ), :z)
    
    #     
    # Rem: There should be a "correct" noise model.
    #      I will look that up at some point.
    
    return z, a′
end;


