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

function proximal_gradient_method(;h = 0.99, nbr_of_iterations = 500, grid = false)
	# import problem data
	Q, q, a, b = problem_data()

	# extract dimensions of x
	dim = length(a)

	# initialize starting points
	x_k = randn(dim)*100

	# extract L
	L = eigmax(Q)
	gamma = 2/L * h

	if grid
		gamma_grid = range(10e-4, stop = 2/L, length = 10)
		residuals = -1 * ones(nbr_of_iterations)
		p = plot()

		for gamma in gamma_grid
			mt = MersenneTwister(123)
			x_k = randn(dim)*100
			for i = 1:nbr_of_iterations
				z = x_k - gamma * grad_quad(x_k,Q,q)
				x_kplus1 = prox_box(z, a, b)
				residuals[i] = norm(x_kplus1 - x_k)
				x_k = x_kplus1
			end
			plot!(p,
				  residuals,
				  yaxis=:log10, 
				  label = round(gamma, digits = 3),
				  legendtitle = "Step Size")
		end
		display(p)
	else
		residuals = -1 * ones(nbr_of_iterations)
		for i = 1:nbr_of_iterations
			z = x_k - gamma * grad_quad(x_k,Q,q)
			x_kplus1 = prox_box(z, a, b)
			residuals[i] = norm(x_kplus1 - x_k)
			x_k = x_kplus1
		end

		plot(residuals, yaxis=:log10)

		return x_k
	end

end


function proximal_dual_gradient_method(;h = 0.99, nbr_of_iterations = 500)
	# import problem data
	Q, q, a, b = problem_data()

	# extract dimensions of x
	dim = length(a)
	# initialize starting points
	y_k = randn(dim)

	# extract L
	L_star = 1/eigmin(Q)

	gamma = 2/L_star * h

	residuals = -1 * ones(nbr_of_iterations)

	for i = 1:nbr_of_iterations
		z = y_k - gamma * grad_quadconj(y_k,Q,q)
		#y_kplus1 = #Här är det fel prox_boxconj(z, a, b)
		residuals[i] = norm(y_kplus1 - y_k)
		y_k = y_kplus1
	end

	plot(residuals, yaxis=:log)
	return y_k
end

x_star = proximal_gradient_method(nbr_of_iterations = 5000)
y_star = proximal_dual_gradient_method(nbr_of_iterations = 5000)
x_star_from_dual = dual2primal(y_star,Q,q)

print("x* = ", norm(x_star))
print("\n")
print("x*_from_dual = ", norm(x_star_from_dual))
