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


function draw_pose!(pose::Pose; l=1., c="b", ax=plt.gca(), zorder=4)
    x, hd = pose
    v = [cos(hd); sin(hd)]
    y = x + l*v
    ax.scatter(x..., c="w", marker="o", edgecolor=c, zorder=zorder)
    ax.plot([x[1]; y[1]], [x[2]; y[2]], c=c, zorder=zorder-1)
end;


"""
    Measurement(z,a)

Sensor measurement containing depth values `z` and
the angles `a` of the simulated laser beams.
"""
struct Measurement <: StructIterator
    z::Vector{Float64} # depth measurements
    a::Vector{Float64} # angles (pre-noise)
end;



"""
    sensor_model(pose::Pose, env::Env, fov, n, a_noise, z_noise, max_z)

Samples a vector of a pseudo lidar measurement
given a pose in an environment.
"""
@gen function sensor_model(pose::Pose, env::Env, fov, n, a_noise, z_noise, max_z)
    x, hd = pose
    I = Diagonal(ones(n))
        
    a0 = fov == 180 ? angles(fov, n+1)[1:end-1] : angles(fov, n+1)
    
    a = @trace(mvnormal(a0 .+ hd, a_noise*I), :a)

    
    z0 = cast(lightcone(a), x, env.segs; max_val=max_z)
    z  = @trace(mvnormal(z0, z_noise .* I), :z)
    
    return Measurement(z, a0)
end;


function draw_sense!(sense::Measurement, pose::Pose; 
                     ax=plt.gca(), s=nothing,c="C1", alpha=.5, zorder=1, rays=true)

    z, a  = sense
    x, hd = pose

    y = euclidean(z, a .+ hd) .+ x'

    if rays
        for i=1:length(a)
            ax.plot([x[1], y[i,1]],[x[2], y[i,2]], c=c, alpha=0.2, zorder=zorder);
        end
    end
    
    ax.scatter(y[:,1], y[:,2], c=c, alpha=alpha, s=s, zorder=zorder);
    
end;