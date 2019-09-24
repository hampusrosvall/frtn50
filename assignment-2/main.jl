using ProximalOperators
using Plots
using LinearAlgebra

include("./problem.jl")

x, y = leastsquares_data()
p = plot()
plot!(p,x,y, seriestype =:scatter)

function min_max_scale(x, x_max, x_min)

    return 2 * (x .- x_min)./(x_max - x_min) .- 1
end

plot!(p,x,min_max_scale(y), seriestype =:scatter)

function expand_x(x, p)
    x_expand = ones(length(x), p + 1)
    for i = 1:p
        x_expand[:, i + 1] = x.^i
    end
    return x_expand
end

function least_squares_gd(X, Y; p = 1, it = 1000, tol = 10e-10, do_plot = false)
    X_scaled = min_max_scale(X, maximum(X), minimum(X))
    X_phi = expand_x(X_scaled, p)
    w = randn(p + 1)
    f = LeastSquares(X_phi, Y)

    gam = 1/eigmax(X_phi * X_phi')
    conv_iter = 0

    for i = 1:it
        w_prev = copy(w)
        gradfw, _ = gradient(f, w)
        w = w .- gam * gradfw

        if norm(w_prev - w) < tol
            conv_iter = i
            break
        end
    end

    if do_plot
        pl = plot()
        plot!(pl, X, Y, seriestype =:scatter)
        plot!(pl, X, X_phi * w)
        display(pl)
        return
    end
    return w, conv_iter
end

least_squares_gd(x, y, p = 10, do_plot = true)

function plot_polynomials(X, Y, p)
    X_grid = range(minimum(X), stop = maximum(X), length = 100)
    X_grid_scaled = min_max_scale(X_grid, maximum(X), minimum(X))
    X_grid_phi = expand_x(X_grid_scaled, p)

    for i = 2:(p + 1)
        w, conv_iter = least_squares_gd(X, Y, p = i-1, it = 1000000, tol = 10e-5)
        print(w, conv_iter)
        print(size(X))
        pl = plot()
        print(X_grid_phi[:,1:i] * w)
        plot!(pl, X, Y, seriestype =:scatter)
        plot!(pl, X, X_grid_phi[:,1:i] * w)
        display(pl)
    end
end

plot_polynomials(x,y,10)
