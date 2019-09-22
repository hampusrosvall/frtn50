using Random

################################################################################
"""
    leastsquares_data()

Returns the data points x_i and y_i for the least squares problem in Hand-In 2.
"""
function leastsquares_data()
	mt = MersenneTwister(0b1101111)
	N = 11

	x = rand(mt,N)
	x = cumsum(x) .- x[1]
	x = x./(x[end]/2) .- 1
	x .*= 2.1

	f(x) =
		1/2*x^4 - 5*x + 1*x^3 + 1*sin(4*x) - 1/(x+2.2) + 4*exp(-abs2(5x)) +
		5*exp(-abs2(5(x - 1.15))) - 3
	
	y = f.(x)
	x .+= 1
	
	return x,y
end


################################################################################
"""
	svm_train()

Returns the training data for SVM problem in Hand-In 2.
"""
function svm_train()
	mt = MersenneTwister(0b11011110)
	N = 500
	x,y = gen_svm_samples(mt,N)
end


"""
Returns the test data sets for SVM problem in Hand-In 2.
"""
function svm_test_1()
	mt = MersenneTwister(0xF0)
	N = 100
	x,y = gen_svm_samples(mt,round(1.8*N))
	i = sortperm(y[1:N])[1:N]
	return x[i], y[i]
end

function svm_test_2()
	mt = MersenneTwister(0xDE)
	N = 100
	x,y = gen_svm_samples(mt,N)
	i = sortperm(y)[1:N]
	return x[i], y[i]
end

function svm_test_3()
	mt = MersenneTwister(0xAC)
	N = 100
	x,y = gen_svm_samples(mt,round(1.8*N))
	i = sortperm(y[end-N+1:end])[1:N]
	return x[i], y[i]
end

function svm_test_4()
	mt = MersenneTwister(0xB12)
	N = 100
	x,y = gen_svm_samples(mt,round(1.8*N))
	i = sortperm(y)[1:N]
	return x[i], y[i]
end


################################################################################
"""
Helper for generating data points
"""
function gen_svm_samples(mt,N)
	n = 4

	xmax = 1
	xmin = -1
	x = [(xmax - xmin)*rand(mt,n) .+ xmin for _ in 1:N]

	nrefs = 40
	mtref = MersenneTwister(0xC0FFEE)
	refs = [(xmax - xmin)*rand(mtref,n) .+ xmin for _ in 1:nrefs]
	refy = [i > nrefs/2 ? 1 : -1 for i = 1:nrefs]
	
	f(x) = sum(i-> exp(-0.5/(.3^2)*norm(x - refs[i])^2)*refy[i], 1:nrefs)

	y = [sign(f(xi)) for xi in x]
	return x,y

end
