using LinearAlgebra

"""
    quad(x,Q,q)

Compute the quadratic

	1/2 x'Qx + q'x

"""
function quad(x,Q,q)
	return 1/2 * x'*Q*x + q'*x
end



"""
    guadconj(y,Q,q)

Compute the convex conjugate of the quadratic

	1/2 x'Qx + q'x

"""
function quadconj(y,Q,q)
	return inv(Q) * (y - q)
end



"""
    box(x,a,b)

Compute the indicator function of for the box contraint

	a <= x <= b

where the inequalites are applied element-wise.
"""
function box(x,a,b)
	return all(a .<= x .<= b) ? 0.0 : Inf
end

"""
    boxconj(y,a,b)

Compute the convex conjugate of the indicator function of for the box contraint

	a <= x <= b

where the inequalites are applied element-wise.
"""
function boxconj(y,a,b)
	S = 0
	for i in eachindex(a)
		if y[i] !=  0
			S += y[i] > 0 ? y[i]*b[i] : y[i] * a[i]
		end
	end
	return S
end

"""
    grad_quad(x,Q,q)

Compute the gradient of the quadratic

	1/2 x'Qx + q'x

"""
function grad_quad(x,Q,q)
	return Q*x + q
end



"""
    grad_quadconj(y,Q,q)

Compute the gradient of the convex conjugate of the quadratic

	1/2 x'Qx + q'x

"""
function grad_quadconj(y,Q,q)
	return inv(Q) * (y - q)
end



"""
    prox_box(x,a,b)

Compute the proximal operator of the indicator function for the box contraint

	a <= x <= b

where the inequalites are applied element-wise.
"""
function prox_box(x,a,b,gamma = 1)
	x_star = zeros(length(x))
	for i in eachindex(x)
		x_star[i] = x[i] <= a[i] ? a[i] : x[i] >= b[i] ? b[i] : x[i]
	end
	return x_star
end



"""
    prox_boxconj(y,a,b)

Compute the proximal operator of the convex conjugate of the indicator function
for the box contraint

	a <= x <= b

where the inequalites are applied element-wise.
"""
function prox_boxconj(y,a,b; gamma = 1)
	return transpose(y) .- prox_box(y, a, b)
end


"""
    dual2primal(y,Q,q,a,b)

Computes the solution to the primal problem for Hand-In 1 given a solution y to
the dual problem.
"""
function dual2primal(y,Q,q; a = 1, b = 1)
	return grad_quadconj(y, Q, q)
end
