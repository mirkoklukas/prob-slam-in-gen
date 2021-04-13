"Beta-like distribution over a bigger interval."
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
    control_prior(mean_speed, max_speed, dhd_noise)

Mindlessly samples a random control vector.
"""
@gen function control_prior(mean_speed, max_speed, dhd_noise)
    s   ~ my_scaled_beta(50., mean_speed, max_speed)
    dhd ~ normal(0, dhd_noise)
    return Control(s, dhd)
end;


"""
    motion_model(pose::Pose, u::Control, hd_noise, x_noise) 

A motion model for an agent that does **not** respect environmental boundaries.
"""
@gen function motion_model(pose::Pose, u::Control, hd_noise, x_noise)    
    hd ~ normal(pose.hd + u.dhd, hd_noise) 
    v = [u.s*cos(hd); u.s*sin(hd)]
    x ~ mvnormal(pose.x + v, Diagonal([x_noise, x_noise]))
    return Pose(x, hd)
end;



function my_incremental_sampling(tr, args, diffs, selection, new_obs, num)
    
    tr,w = update(tr, args, diffs, new_obs)
    

    diffs = changes(zeros(length(args)))
    
    trs = []
    ws = []
    for i=1:num
        tr′, w′ = regenerate(tr, args, diffs, selection)
        push!(trs, tr′)
        push!(ws, w′)
    end
    return trs, ws
end;



sensor_mixture = HeterogeneousMixture([normal; uniform])

struct MultivariateSensor <: Gen.Distribution{Vector{Float64}} end
const mvsensor = MultivariateSensor()

function Gen.logpdf(::MultivariateSensor, z::AbstractArray{Float64,1}, 
                    mu::AbstractArray{Float64,1}, sig::Float64, max_val::Float64, w::Float64)
    
    n = length(mu)
    f = map((i) -> Gen.logpdf(sensor_mixture, z[i], [1.0-w, w], mu[i], sig, 0.0, max_val), 1:n)
    f = sum(f)
    return f
end

function Gen.random(::MultivariateSensor, 
                    mu::AbstractArray{Float64,1}, 
                    sig::Float64, 
                    max_val::Float64, 
                    w::Float64)
    n = length(mu)
    z = [sensor_mixture([1.0-w, w], mu[i], sig, 0.0, max_val) for i=1:n]
    return z
end

(::MultivariateSensor)(mu::AbstractArray{Float64,1}, 
                       sig::Float64, 
                       max_val::Float64, 
                       w::Float64) = Gen.random(MultivariateSensor(), mu, sig, max_val, w)

Gen.has_output_grad(::MultivariateSensor) = false;
Gen.has_argument_grads(::MultivariateSensor) = (false,false,false,false);



@gen (static) function sensor_model_loosened(pose::Pose, env::Env, fov, n, z_noise, max_z, drop_out)
    x, hd = pose
    
    a0 = angles(fov, n)
    a = a0 .+ hd 
    z0 = cast(lightcone(a), x, env.segs; max_val=max_z)
    z  = @trace(mvsensor(z0, z_noise, max_z, drop_out), :z)
    
    return Measurement(z, a0)
end;



@gen (static) function new_localization_kernel(t::Int, 
                          pose::Pose, 
                          env::Env, 
                          control_args,
                          transition_args,
                          sensor_args, drop_out)

    mean_speed, max_speed, dhd_noise = control_args
    hd_noise, x_noise                = transition_args
    fov, n, z_noise, max_z  = sensor_args
    
    u     = @trace(control_prior(mean_speed, max_speed, dhd_noise), :u)
    pose  = @trace(motion_model(pose, u, hd_noise, x_noise), :pose)
    s     = @trace(sensor_model_loosened(pose, env, fov, n, z_noise, max_z, drop_out), :sense)
    
    return pose
end

new_loc_chain = Gen.Unfold(new_localization_kernel);
Gen.load_generated_functions();