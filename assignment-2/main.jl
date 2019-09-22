using ProximalOperators
using Plots
using LinearAlgebra

include("./problem.jl")

x, y = leastsquares_data()
p = plot()
plot!(p,x,y, seriestype =:scatter)

function min_max_scale(y)
    y_max = maximum(y)
    y_min = minimum(y)

    return 2 * (y .- y_min)./(y_max - y_min) .- 1
end

plot!(p,x,min_max_scale(y), seriestype =:scatter)

function least_squares_prox(x, y_unscaled; it = 1000, tol = 10e-6)
    p = size(x[1,:])
    w = randn(p)
    y = min_max_scale(y_unscaled)
    f = SqrNormL2()

    gam = 1/eigmax(x * x')

    for i = 1:1000
        z = w .- gam.*(x'*(x.*w - y))
        w_prev = w
        prox!(w, f, z, gam)

        #OM VI VILL KÖRA MED TOL ISTÄLLET
        #if norm(w_prev - w) < tol
        #    print(i)
        #    break
        #end
    end
    return w
end

w = least_squares_prox(x,y, it = 10000000)

p = plot()
plot!(p, x, y_scaled, seriestype =:scatter)
plot!(p, x, x.*w)

function expand_x(x, p)
    x_expand = ones(length(x), p + 1)
    for i = 2:(p + 1)
        x_expand[:, i] = x.^(i - 1)
    end
    return x_expand
end

function plot_ploynomials(x,y,p)
    x_expand = expand_x(x, p)
    y_scaled = min_max_scale(y)
    
    p = Plot()
    plot!(p, x, y_scaled, seriestype =:scatter)
    for i = 1:(p + 1)
        x_slize = x_expand[:,1:i]
        w = least_squares_prox(x_slize, y)
        plot!(p, x, x*w')
    end
end
