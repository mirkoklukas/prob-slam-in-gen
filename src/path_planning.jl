import Gen2DAgentMotion: plan_path, refine_path, Path, Scene, 
    Bounds, Wall, Point, PlannerParams, RRTTree, simplify_path, plan_and_optimize_path

Scene(env::Env) = Scene(Bounds(env.bounds'...),[Wall(Point(s[1:2]...),Point(s[3:4]...)) for s in eachrow(env.segs)] )


function random_position_inside(env)
    while true
        x  = mvuniform(env.bounds)
        hd = uniform(0,2π)
        if is_inside(x, env.segs)
            return (x=x, hd=hd)
        end
    end
end


function decompose(path::Array{Point,1})
    x   = vcat([[p.x p.y] for p in path]...)
    dx  = x[2:end,:] - x[1:end-1,:]
    hd  = [atan(v[2],v[1]) for v in eachrow(dx)]
    s   = sqrt.(sum(dx.^2, dims=2)[:,1])
    dhd = (hd[2:end] - hd[1:end-1]).%2π    
    return x, hd, s, dhd 
end

decompose(path::Path) = decompose(path.points)


function draw_tree!(tree::RRTTree; ax=plt.gca(), c="black",zorder=nothing, alpha=0.5)
    for node in 1:tree.num_nodes
        a = Point(tree.confs[1,node], tree.confs[2,node])
        parent = tree.parents[node]
        if parent == 0
            continue
        end
        b = Point(tree.confs[1,parent],tree.confs[2,parent])
        ax.plot([a.x, b.x], [a.y, b.y], color=c, zorder=zorder, alpha=alpha)
    end
end;



function get_random_path(env, 
                         params = PlannerParams(2000, 1., .01, 1000, 1.0, 0.2); 
                         simplify=false)
    p = random_position_inside(env)
    q = random_position_inside(env)


    while true
        global path,tree;
        path, tree = plan_path(Point(p.x...), Point(q.x...), Scene(env), params);

        if path != nothing
            if simplify
                path = simplify_path(Scene(env), path; spacing=1.)
            end
            break
        else
            p = random_position_inside(env)
            q = random_position_inside(env)
        end
    end
    
    return path
end;



