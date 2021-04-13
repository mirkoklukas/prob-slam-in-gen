"Beta-like distribution over a bigger interval with prescribed mean."
@dist function my_scaled_beta(a, mean, scale)
    mean_beta = mean/scale
    b = a/mean_beta - a
    scale*beta(a,b);
end;


"""
    Control

Contains the controls of the agent. 
"""
struct Control <: StructIterator
    "speed"
    s::Float64 
    "change of head direction"
    dhd::Float64
end;


"""
    Control(s, dhd) = control_prior(mean_speed, max_speed, sigma_dhd)

Mindlessly samples a random control vector.

Choices:
```
    |
    +-- :s : Float64
    |
    +-- :dhd : Float64
```
"""
@gen function control_model(mean_speed, max_speed, dhd_noise)
    s   ~ my_scaled_beta(4., mean_speed, max_speed)
    dhd ~ normal(0, dhd_noise)
    return Control(s, dhd)
end;

# Rem: The control model could depend on 
# ---  the current observations and the environment 


"""
    motion_model(pose::Pose, u::Control, hd_noise, x_noise) 

A mindless motion model for an agent that does **not** respect environmental boundaries.
That means the map argument is ignored.
"""
@gen function motion_model(pose::Pose, u::Control, hd_noise, x_noise)    
    hd ~ normal(pose.hd + u.dhd, hd_noise) 
    v = [u.s*cos(hd); u.s*sin(hd)]
    x ~ mvnormal(pose.x + v, Diagonal([x_noise, x_noise]))
    return Pose(x, hd)
end;

# Rem: The motion model could depend on the map/environment - one 
# ---  might want to check if the sampled move is in fact a valid one (see the other notebooks).


"""
    loc_kernel(t::Int, 
                state::SLAMState, 
                M::Map, 
                control_args,
                motion_args,
                sensor_args)

MC kernel for localization on a given map.
"""
@gen (static) function loc_kernel(t::Int, 
                          pose::Pose, # State of kernel
                          M::Map,
                          control_args,
                          motion_args,
                          sensor_args)
    
    mean_speed, max_speed, dhd_noise = control_args
    hd_noise, x_noise = motion_args
    fov, n, max_val, noise, drop_out = sensor_args
    
    u     = @trace(control_model(mean_speed, max_speed, dhd_noise), :u)
    pose  = @trace(motion_model(pose, u, hd_noise, x_noise), :pose)
    sense = @trace(sensor_model(M, pose,fov, n, max_val, noise, drop_out), :sense)

    return pose
end

loc_chain = Gen.Unfold(loc_kernel);

@gen (static) function slam_model(T::Int, 
                         map_args,
                         control_args,
                         motion_args,
                         sensor_args)
    
    map_size, res, occ_prior = map_args
    fov, n, max_val, noise, drop_out = sensor_args
    
    # Initial state of the MC
    m  = @trace(map_prior(map_size, res, occ_prior), :M)
    p  = @trace(pose_prior(m), :pose)
    sense = @trace(sensor_model(m, p, fov, n, max_val, noise, drop_out), :sense)
    
    # Unfolding the chain
    chain = @trace(loc_chain(T, p, m, control_args, motion_args, sensor_args), :chain)
    
    return chain
end;

Gen.load_generated_functions()