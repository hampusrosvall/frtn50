using ProximalOperators
using Plots
using LinearAlgebra
using Statistics

include("./problem.jl")

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

function create_kernel_vector(X_test, X, sigma)
    K = zeros(length(X))
    for i in eachindex(X)
        K[i] = exp(-(1/(2*sigma^2)) * norm(X[i] - X_test)^2)
    end
    return K
end

function generate_beta(beta_type, k; mu = 1, gamma = 1)
    if beta_type == 1
        return (k - 2) / (k + 1)
    elseif beta_type == 3
        return (1 - sqrt(mu * gamma)) / (1 + sqrt(mu * gamma))
    else
        throw(ArgumentError("beta_type should be 1 or 3"))
    end

end

function svm_dual_solver(X, Y; lam = 0.0001, it = 100000, sigma = 0.5, beta_type = 0, cd = false)
    mt = MersenneTwister(123)
    K = create_kernel(X, sigma = sigma)
    Q = (1/lam) * diagm(Y) * K * diagm(Y)
    N = length(X)
    gam = 1 / eigmax(Q)

    # Initialize functions
    h = HingeLoss(ones(N), 1/N)
    h_conj = Conjugate(h)
    v = randn(mt, length(X))
    q = zeros(N)
    g_conj = Quadratic(Q, q)
    v_iter = zeros((it, length(X)))

    if cd
        h_j = HingeLoss(ones(1), 1)
        h_jconj = Conjugate(h_j)
        for i = 1:it
            # Perform coordinate gradient descent step
            j = rand(1:length(v))

            # gradient of quadratic function
            #gradgv, _ = gradient(g_conj, v)
            gradgv = Q[j, :]' * v

            # Perform gradient descent step w.r.t coordinate j
            v_jgd = v[j] - gam * gradgv

            # calculate new point v
            v_j, _ = prox(h_jconj, [v_jgd], gam)

            v[j] = v_j[1]

            v_iter[i, :] = v
        end
    else
        if beta_type > 0
            v_prev = v
            mu = eigmin(Q)
            for i = 1:(it * N)
                # Generate Beta_k
                B_k = generate_beta(beta_type, i, mu = mu, gamma = gam)

                # Perform extrapolation step
                v_extra = v + B_k * (v - v_prev)

                # Cache old point
                v_prev = v

                # Calculate gradient
                gradgv, _ = gradient(g_conj, v_extra)

                # update dual parameter v
                v = v - gam * gradgv

                # Update point
                v, _ = prox(h_conj, v, gam)
                v_iter[i, :] = v
            end
        else
            for i = 1:it
                # Perform proximal gradient descent step

                # gradient of quadratic function
                gradgv, _ = gradient(g_conj, v)

                # update dual parameter mu
                v = v - gam * gradgv

                # calculate new point v
                v, _ = prox(h_conj, v, gam)

                if i % 500
                    v_iter[Int(i/N), :] = v
                end
            end
        end
    end

    return v, v_iter
end

function predict(X_test, v, Y, X; lambda = 1, sigma = 1)
    Y_pred = zeros(length(X_test))

    for i in eachindex(X_test)
        K = create_kernel_vector(X_test[i], X, sigma)
        Y_pred[i] = sign((-1/lambda) * transpose(v) * diagm(Y) * K)
    end
    return Y_pred
end

function point_generator()
    X, Y = svm_train()
    print("Solving dual problem to high precision..\n")
    v_star, v_iter = svm_dual_solver(X, Y)
    print("Solving dual problem to for beta_1..\n")
    _, v1_iter = svm_dual_solver(X, Y, beta_type = 1)
    print("Solving dual problem to for beta_3..\n")
    _, v3_iter = svm_dual_solver(X, Y, beta_type = 3)

    return v_iter, v1_iter, v3_iter, v_star
end

sigma = 0.5
lambda = 0.0001

# Initialize training data
X, Y = svm_train()

# Solve the dual problem
v = svm_dual_solver(X, Y, lam = lambda, sigma = sigma, beta_type = 3)

v_iter, v1_iter, v3_iter, v_star = iteration_generator()

v1_iter_norm = [norm(v1_iter[i, :] .- v_star) for i = 1:length(v1_iter[:, 1])]
v3_iter_norm = [norm(v3_iter[i, :] .- v_star) for i = 1:length(v3_iter[:, 1])]

pl = plot(yaxis=:log10)
plot!(pl, v1_iter_norm, label = "Beta_1")
plot!(pl, v3_iter_norm, label = "Beta_3")
display(pl)
