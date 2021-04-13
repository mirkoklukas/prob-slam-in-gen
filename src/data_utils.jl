function set_constr!(ch, pose::Pose, t::Int64=0)
    if t==0
        ch[:pose => :x]  = pose.x
        ch[:pose => :hd] = mod(pose.hd, 2π)
    else
        ch[:chain => t => :pose => :x] = pose.x
        ch[:chain => t => :pose => :hd] = pose.hd
    end
end

function set_constr!(ch, u::Control, t::Int64)
    ch[:chain => t => :u => :s] = u.s
    ch[:chain => t => :u => :dhd] = u.dhd
end

function set_constr!(ch, sense::Measurement, t::Int64=0)
    if t==0
        ch[:sense => :z] = sense.z
    else
        ch[:chain => t => :sense => :z] = sense.z
    end
end

function extract_poses(data)
    x   = data["x"]
    hd  = data["hd"]
    T   = length(hd)
    p = [Pose(x[t,:], mod(hd[t],2π)) for t=1:T]
    
    return p
end

function extract_controls(data)
    s   = data["s"]
    dhd = data["dhd"]
    T   = length(dhd)
    u = [Control(s[t], dhd[t]) for t=1:T]
    
    return u
end



function extract_sensor_args(data)
    fov = data["fov"]
    n = data["n"]
    max_val = data["max_val"]
    return (fov=fov, n=n, max_val=max_val)
end;


function extract_measurements(data)
    z = data["z"]
    a = data["a"]
    sen = [Measurement(z[t], a[t]) for t=1:length(a)]
    return sen
end;


