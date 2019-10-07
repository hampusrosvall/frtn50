using ProximalOperators
using Plots
using LinearAlgebra
using Statistics

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

    gam = 1/eigmax(X_phi' * X_phi)
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
        w, conv_iter = least_squares_gd(X, Y, p = i-1, it = 1000000, tol = 10e-15)
        pl = plot()
        plot!(pl, X, Y, seriestype =:scatter)
        plot!(pl, X_grid, X_grid_phi[:,1:i] * w)
        display(pl)
    end
end

plot_polynomials(x,y,15)

function least_squares_reg(X, Y; q = 2, p = 10, lam = 0.1, it = 10000, do_plot = false, scale = true)
    if scale
       X_grid = range(minimum(X), stop = maximum(X), length = 100)
       X_grid_scaled = min_max_scale(X_grid, maximum(X), minimum(X))
       X_grid_phi = expand_x(X_grid_scaled, p)
       X_scaled = min_max_scale(X, maximum(X), minimum(X))
       X_phi = expand_x(X_scaled, p)
   else
       X_grid = range(minimum(X), stop = maximum(X), length = 100)
       X_grid_phi = expand_x(X_grid, p)
       X_phi = expand_x(X, p)
   end
    # Initializing functions
    f = LeastSquares(X_phi, Y)

    gam = 1/eigmax(X_phi' * X_phi)

    pl = plot()
    plot!(pl, X, Y, seriestype =:scatter, label = 'Data')

    # Proximal gradient descent
    for l in lam
        g = q == 2 ? NormL2(l) : NormL1(l)
        # Initialize weights w
        w = randn(p + 1)
        for i = 1:it
            # gradient of least squares
            gradfw,_ = gradient(f, w)
            w_step = w - gam * gradfw
            w, _ = prox(g, w_step, gam)
        end
        if do_plot
            plot!(pl, X_grid, X_grid_phi * w, label = l)
        end
    end
    display(pl)
end

lam_grid = range(0.001, stop = 10, length = 5)
least_squares_reg(x, y, lam = lam_grid, do_plot = true)

function create_kernel(X; sigma = 1)
    dim = length(X)
    K = zeros(dim, dim)
    for (i, x_i) in enumerate(X)
        for (j, x_j) in enumerate(X)
            exponent = -(1 / (2*sigma^2)) * norm(x_i - x_j)^2
            K[i, j] = exp(exponent)
        end
    end
    return K
end

function svm_dual_solver(X, Y; lam = 0.01, it = 100000, sigma = 1)
    K = create_kernel(X, sigma = sigma)
    Q = (1/lam) * diagm(Y) * K * diagm(Y)
    N = length(X)
    gam = 1 / eigmax(Q)

    # Initialize functions
    h = HingeLoss(ones(N), 1/N)
    h_conj = Conjugate(h)
    v = randn(length(X))
    q = zeros(N)
    g_conj = Quadratic(Q, q)

    for i = 1:it
        # Perform proximal gradient descent step

        # gradient of quadratic function
        gradgv, _ = gradient(g_conj, v)

        # update dual parameter mu
        v = v - gam * gradgv

        # calculate new point v
        v, _ = prox(h_conj, v, gam)
    end
    return v
end

function create_kernel_vector(X_test, X, sigma)
    K = zeros(length(X))
    for i in eachindex(X)
        K[i] = exp(-(1/(2*sigma^2)) * norm(X[i] - X_test)^2)
    end
    return K
end


function predict(X_test, v, Y, X; lambda = 1, sigma = 1)
    Y_pred = zeros(length(X_test))

    for i in eachindex(X_test)
        K = create_kernel_vector(X_test[i], X, sigma)
        Y_pred[i] = sign((-1/lambda) * transpose(v) * diagm(Y) * K)
    end
    return Y_pred
end

sigma = 0.3
# Initialize training data
X, Y = svm_train()

# Solve the dual problem
v = svm_dual_solver(X, Y, lam = 0.01, sigma = sigma)

function grid_search(lambda_grid, sigma_grid)
    X, Y = svm_train()
    X_test, Y_test = svm_test_1()
    print("============= GRID SEARCH =============\n")
    for lam in lambda_grid
        for sigma in sigma_grid
            # Solve the dual problem
            v = svm_dual_solver(X, Y, lam = lam, sigma = sigma)

            # Make predictions on test data
            Y_pred_test = predict(X_test, v, Y, X, lambda = lam, sigma = sigma)

            # Calculate error rate on test data
            error_test = mean(Y_pred_test .!= Y_test)

            # Predict on training data
            Y_pred_train = predict(X, v, Y, X, lambda = lam, sigma = sigma)

            # Calculate error rate on train data
            error_train = mean(Y_pred_train .!= Y)

            # Print outs
            print("Lambda: ", lam, " Sigma: ", sigma, " Test error rate: ", error_test,
                            " Train error rate: ", error_train, "\n")
        end
    end
end

lambda_grid = [0.1, 0.01, 0.001, 0.0001]
sigma_grid = [1, 0.5, 0.25]

grid_search(lambda_grid, sigma_grid)

function all_test_sets(;lam = 0.01, sigma = 0.25, naive_clf = false)
    # Load training data
    X, Y = svm_train()

    # Load all test data
    X_test_1, Y_test_1 = svm_test_1()
    X_test_2, Y_test_2 = svm_test_2()
    X_test_3, Y_test_3 = svm_test_3()
    X_test_4, Y_test_4 = svm_test_4()

    # Load all test data into array of tuples
    test_data = [
    (X_test_1, Y_test_1),
    (X_test_2, Y_test_2),
    (X_test_3, Y_test_3),
    (X_test_4, Y_test_4)
    ]

    # Train model
    v = svm_dual_solver(X, Y, lam = lam, sigma = sigma, it = 100000)

    # Predict on test sets
    print("============== PRECICT ON TEST DATA ==============\n")
    print("Lambda: ", lam, " Sigma: ", sigma, "\n")
    Y_pred = predict(X, v, Y, X, lambda = lam, sigma = sigma)
    error_rate = mean(Y_pred .!= Y)
    print("Error rate on train set : ", error_rate, "\n")

    for (i, (X_test, Y_test)) in enumerate(test_data)
        Y_pred = predict(X_test, v, Y, X, lambda = lam, sigma = sigma)
        error_rate = mean(Y_pred .!= Y_test)
        print("Error rate on test set ", i, ": ", error_rate, "\n")

        if naive_clf
            error_rate_pos = mean(Y_test .!= 1 .* ones(length(Y_test)))
            error_rate_neg = mean(Y_test .!= -1 .* ones(length(Y_test)))
            print("Pos clf: ", error_rate_pos, "\n")
            print("Neg clf: ", error_rate_neg, "\n")
        end
    end

end

all_test_sets(lam = 0.1, sigma = 2, naive_clf = true)
all_test_sets(lam = 0.001, sigma = 0.5, naive_clf = true)
all_test_sets(lam = 0.00001, sigma = 0.25)

function k_fold_svm()
    X, Y = svm_train()

    # shuffling data
    idxs = randperm(length(X))

    # extract shuffled data
    X = X[idxs]
    Y = Y[idxs]

    folds = 10
    valid_size = Integer(length(X) / folds)
    error_rate = 0

    for i = 1:folds
        # Slice out validation and train set
        start = (i - 1) * valid_size + 1
        stop = i * valid_size
        valid_idxs = start:stop
        train_idxs = setdiff(1:length(X), valid_idxs)
        Y_valid = Y[valid_idxs]
        Y_train = Y[train_idxs]
        X_valid = X[valid_idxs]
        X_train = X[train_idxs]

        # Train the model
        v = svm_dual_solver(X_train, Y_train, lam = 0.01, sigma = 0.25)

        # Predict on validation set
        Y_pred = predict(X_valid, v, Y_train, X_train, lambda = 0.01, sigma = 0.25)

        error_rate += mean(Y_pred .!= Y_valid)
    end

    return error_rate / folds
end

k_fold_svm()

function holdout_svm(;holdout_size = 0.25)
    X, Y = svm_train()

    n_valid = Integer(length(X)*holdout_size)

    # shuffling data
    idxs = randperm(length(X))

    # extract shuffled data
    X = X[idxs]
    Y = Y[idxs]

    # Set aside test set
    X_valid = X[1:n_valid]
    Y_valid = Y[1:n_valid]

    X_train = X[(n_valid+1):end]
    Y_train = Y[(n_valid+1):end]

    # Train the model
    v = svm_dual_solver(X_train, Y_train, lam = 0.01, sigma = 0.25)

    Y_pred = predict(X_valid, v, Y_train, X_train, lambda = 0.01, sigma = 0.25)
    Y_pred_train = predict(X_train, v, Y_train, X_train, lambda = 0.01, sigma = 0.25)

    valid_error = mean(Y_pred .!= Y_valid)

    return valid_error

end

holdout_svm()
