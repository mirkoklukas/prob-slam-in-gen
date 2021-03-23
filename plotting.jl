
function draw_segs!(segs; c="black", ax=plt.gca(), alpha=1.)
    ax.plot(segs[:,[1, 3]]', segs[:,[2, 4]]', c=c, alpha=alpha, linewidth=1, zorder=1); 
end


function draw_env!(env; ax = plt.gca(), title="", alpha=1., zorder=1, 
                   wall="black", grass="lightgray", floor="w" )
    v = vcat(env.verts, env.verts[[1],:])
    

    box = env.bounds[:,:]
    box[:,1] .-= 2.
    box[:,2] .+= 2.
    box = hcat([box[:,1], [box[1,2], box[2,1]], [box[1,2], box[2,2]],  [box[1,1], box[2,2]], box[:,1]]...)

    ax.set_title(title)
    ax.fill(box[1,:],box[2,:], c=grass, alpha=1., zorder=-1);
    ax.fill(v[:,1], v[:,2], c=floor, linewidth=0, zorder=-1);
    ax.plot(v[:,1], v[:,2], c=wall, alpha=alpha, linewidth=1, zorder=zorder);
end;


function draw_pose!(x,r; len=1.,m="o", ec="None", c="C0", ax=plt.gca(), zorder=4)
    v = [cos(r); sin(r)]
    y = x + len*v
    ax.scatter(x..., c=c, marker=m, zorder=zorder, edgecolor=ec)
    ax.plot([x[1]; y[1]], [x[2]; y[2]], c=c, zorder=zorder)
end;


function draw_sense!(x, rot, z, sensor_args; c="C1", ax=plt.gca(), alpha=0.5)
    θs = angles(sensor_args.fov, sensor_args.num_rays) .+ rot
    hits = z.*hcat(cos.(θs), sin.(θs)) .+ x'
    valid = z .< sensor_args.max_depth - 5
    ax.scatter(hits[valid,1], hits[valid,2], c=c, marker=".", alpha=alpha)
end