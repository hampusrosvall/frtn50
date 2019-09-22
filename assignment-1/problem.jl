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

""" Task 6
	proximal_gradient_method()

Returns the solution for the primal problem.
Args:
	h 				  : proportion of the upperbound stepsize
	nbr_of_iterations : numer of steps we take
	grid 			  : if true plot convergece for different step sizes
	do_plot 		  : if true plots the convergence
"""

function proximal_gradient_method(;h = 0.99, nbr_of_iterations = 500, grid = false, do_plot = false)
	# import problem data
	Q, q, a, b = problem_data()
	# extract dimensions of x
	dim = length(a)
	# initialize starting points
	x_k = randn(dim)*100
	# extract L
	L = eigmax(Q)
	gamma = 2/L * h

	#Check convergence for different step sizes
	if grid
		#Create different step sizes
		gamma_grid = range(10e-4, stop = 2/L, length = 10)
		residuals = -1 * ones(nbr_of_iterations)
		p = plot()

		for gamma in gamma_grid
			#Set the starting point to the same value
			mt = MersenneTwister(123)
			x_k = randn(dim)*100
			#Optimization loop
			for i = 1:nbr_of_iterations
				z = x_k - gamma * grad_quad(x_k,Q,q)
				x_kplus1 = prox_box(z, a, b)
				#for plotting
				residuals[i] = norm(x_kplus1 - x_k)
				x_k = x_kplus1
			end
			#Plot in same plot
			plot!(p,
				  residuals,
				  yaxis=:log10,
				  label = round(gamma, digits = 3),
				  legendtitle = "Step Size")
		end
		display(p)
	else
		residuals = -1 * ones(nbr_of_iterations)
		#Optimization loop
		for i = 1:nbr_of_iterations
			z = x_k - gamma * grad_quad(x_k,Q,q)
			x_kplus1 = prox_box(z, a, b)
			#for plotting
			residuals[i] = norm(x_kplus1 - x_k)
			x_k = x_kplus1
		end
		if do_plot
			plot(residuals, yaxis=:log10)
		else
			return x_k
		end
	end

end

function prox_start_point(;h = 0.99, nbr_of_iterations = 2000)
	# import problem data
	Q, q, a, b = problem_data()

	# extract dimensions of x
	dim = length(a)

	# initialize grid of starting points
	scalars = range(1, stop = 1000, length = 5)

	# extract L
	L = eigmax(Q)
	gamma = 2/L * h
	p = plot()

	residuals = -1 * ones(nbr_of_iterations)
	p = plot()

	for scalar in scalars
		x_k = randn(dim)*scalar
		for i = 1:nbr_of_iterations
			z = x_k - gamma * grad_quad(x_k,Q,q)
			x_kplus1 = prox_box(z, a, b)
			residuals[i] = norm(x_kplus1 - x_k)
			x_k = x_kplus1
		end
		plot!(p,
			  residuals,
			  yaxis=:log10,
			  label = floor(scalar, digits = 0),
			  legendtitle = "Starting point scale",
			  xlabel = "# iterations",
			  ylabel = "Norm of residual",
			  title = "Convergence rates for different starting points")
	end
	display(p)
	savefig("starting.png")
end

""" Task 7
	proximal_dual_gradient_method()

Returns the solution for the dual problem.
Args:
	h 				  : proportion of the upperbound stepsize
	nbr_of_iterations : numer of steps we take
	do_plot 		  : if true plots the convergence
"""
function proximal_dual_gradient_method(;h = 0.99, nbr_of_iterations = 500, do_plot = false, plot_f_x = false, plot_f_i_x = false)
	# import problem data
	Q, q, a, b = problem_data()
	# extract dimensions of x
	dim = length(a)
	# initialize starting points
	y_k = randn(dim) * 100
	# extract L
	L_star = 1/eigmin(Q)
	#Step Size
	gamma = 2/L_star * h
	#List to store the norm for plotting
	residuals = -1 * ones(nbr_of_iterations)

	#Optimization loop
	for i = 1:nbr_of_iterations
		z = y_k - gamma * grad_quadconj(y_k,Q,q)
		y_kplus1 = -prox_boxconj(-z, a, b, gamma = gamma)
		#Plotting either:
		# f(x^k) recovered from the dual
		# f(x^k) + g(x^k) recovered from the dual
		# Norm of the residuals
		if plot_f_x
			x = dual2primal(y_kplus1,Q,q)
			residuals[i] = quad(x,Q,q)
		elseif plot_f_i_x
			x = dual2primal(y_kplus1,Q,q)
			residuals[i] = quad(x,Q,q) + box(x,a,b)
		else
			residuals[i] = norm(y_kplus1 - y_k)
		end

		y_k = y_kplus1
	end

	if do_plot
		plot(residuals, yaxis=:log)
		#plot(residuals, yaxis=:log, legend = false, title="Evolution of f(x) over iterations", xlabel = "# iterations k", ylabel = "f(x)")
	else
		return y_k
	end
end

function check_if_dual_in_S()
	Q, q, a, b = problem_data()
	for i in [10,50,100,500,1000,5000,10000,50000,100000, 500000]
		y_star = proximal_dual_gradient_method(nbr_of_iterations = i)
		x_star_from_dual = dual2primal(y_star,Q,q)
		in_S = box(x_star_from_dual, a, b)
		if in_S == 0
			print("Inside for ", i, " iterations\n")
		else
			print("Outside for ", i, " iterations\n")
		end
	end
end



function main()

	Q, q, a, b = problem_data()
	n_itrs = 500000
	x_star = proximal_gradient_method(nbr_of_iterations = n_itrs)
	y_star = proximal_dual_gradient_method(nbr_of_iterations = n_itrs)
	x_star_from_dual = dual2primal(y_star,Q,q)

	print("Primal x* = ", norm(x_star))
	print("\n")
	print("Dual x* = ", norm(x_star_from_dual))
	print("\n \n")
	print("The norm difference in x* is: \n", norm(x_star - x_star_from_dual))

end

main()
