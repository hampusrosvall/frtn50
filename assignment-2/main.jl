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

function expand_x(x, p)
    x_expand = ones(length(x), p + 1)
    for i = 1:p
        x_expand[:, i + 1] = x.^i
    end
    return x_expand
end

function least_squares_gd(x_unexpanded, y_unscaled; p = 1, it = 1000, tol = 10e-10, do_plot = false)
    X = expand_x(x_unexpanded, p)
    w = randn(p+1)
    Y = min_max_scale(y_unscaled)
    f = LeastSquares(X, Y)

    gam = 1/eigmax(X * X')
    conv_iter = 0

    for i = 1:it
        w_prev = copy(w)
        gradfw, _ = gradient(f, w)
        w = w - gam * gradfw

        if norm(w_prev - w) < tol
            conv_iter = i
            break
        end
    end

    if do_plot
        pl = plot()
        plot!(pl, x_unexpanded, Y, seriestype =:scatter)
        plot!(pl, x_unexpanded, X * w)
        display(pl)
        return
    end
    return w, conv_iter
end

least_squares_gd(x, y, p = 4, do_plot = true)

function plot_polynomials(x, y, p)
    y_scaled = min_max_scale(y)

    x_grid = range(minimum(x), stop = maximum(x), length = 100)
    x_grid_expand = expand_x(x_grid, p)
    #pl = plot()
    #plot!(pl, x, y_scaled, seriestype =:scatter, ylims = (-1.2,1.2))

    for i = 2:(p + 1)
        w, conv_iter = least_squares_gd(x, y, p = i-1, tol = 10e-17)
        #plot!(pl, x_grid, x_grid_expand[:,1:i]*w)
        pl = plot()
        plot!(pl, x, y_scaled, seriestype =:scatter, ylims = (-1.2,1.2))
        y_pred = x_grid_expand[:,1:i] * w
        plot!(pl, x_grid, x_grid_expand[:,1:i] * w)
        display(pl)
    end
end

plot_polynomials(x,y,10)
