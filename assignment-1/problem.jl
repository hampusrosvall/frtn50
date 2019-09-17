using Random
using Plots

include("./functions.jl")

"""
    problem_data()

Returns the Q, q, a, and b matrix/vectors that defines the problem in Hand-In 1.

"""
function problem_data()
	mt = MersenneTwister(123)

	n = 20

	Qv = randn(mt,n,n)
	Q = Qv'*Qv
	q = randn(mt,n)

	a = -rand(mt,n)
	b = rand(mt,n)

	return Q,q,a,b
end

"""
	proximal_gradient_method()

Returns the minimizer x_kplus1 given the
"""

function proximal_gradient_method(;h = 0.99, nbr_of_iterations = 500; grid = false)
	# import problem data
	Q, q, a, b = problem_data()

	# extract dimensions of x
	dim = length(a)

	# initialize starting points
	x_k = randn(dim)

	# extract L
	L = eigmax(transpose(Q) * Q)

	gamma = 2/L * h

	if !grid
		residuals = -1 * ones(nbr_of_iterations)

		for i = 1:nbr_of_iterations
			z = x_k - gamma * grad_quad(x_k,Q,q)
			x_kplus1 = prox_box(z, a, b)
			residuals[i] = norm(x_kplus1 - x_k)
			x_k = x_kplus1
		end

		plot(residuals, yaxis=:log)
	else:
		gamma_grid = range(0.001, gamma, 10)
		residuals = -1 * ones(nbr_of_iterations)

		for gamma in eachindex(gamma_grid)
			for i = 1:nbr_of_iterations
				z = x_k - gamma * grad_quad(x_k,Q,q)
				x_kplus1 = prox_box(z, a, b)
				residuals[i] = norm(x_kplus1 - x_k)
				x_k = x_kplus1
			end

			plot!(residuals, yaxis=:log)
end
