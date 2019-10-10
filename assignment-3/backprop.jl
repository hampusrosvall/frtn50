using LinearAlgebra, Statistics, Random

# We define some useful activation functions
sigmoid(x) = exp(x)/(1 + exp(x))
relu(x) = max.(x)
leakyrelu(x) = max.(x, 0.2 * x)

# And methods to calculate their derivatives
derivative(f::typeof(sigmoid), x::Float64) = sigmoid(x)*(1-sigmoid(x))
derivative(f::typeof(identity), x::Float64) = one(x)
derivative(f::typeof(relu), x::Float64) = x < 0 ? 0 : 1
derivative(f::typeof(leakyrelu), x::Float64) = x < 0 ? 0.2 : 1

# Astract type, all layers will be a subtype of `Layer`
abstract type Layer{T} end

""" Dense layer for `σ(W*z+b)`,
    stores the intermediary value z as well as the output, gradients and δ"""
struct Dense{T, F<:Function} <: Layer{T}
    W::Matrix{T}
    b::Vector{T}
    σ::F
    x::Vector{T}    # W*z+b
    out::Vector{T}  # σ(W*z+b)
    ∂W::Matrix{T}   # ∂J/dW
    ∇b::Vector{T}   # (∂J/db)ᵀ
    δ::Vector{T}    # dJ/dz
end

""" layer = Dense(nout, nin, σ::F=sigmoid, W0 = 1.0, Wstd = 0.1, b0=0.0, bstd = 0.1)
    Dense layer for `σ(W*x+b)` with nout outputs and nin inputs, with activation function σ.
    `W0, Wstd, b0, bstd` adjusts the mean and standard deviation of the initial weights. """
function Dense(nout, nin, σ::F=sigmoid, W0 = 1.0, Wstd = 0.1, b0=0.0, bstd = 0.1) where F
    W = W0/nin/nout .+ Wstd/nin/nout .* randn(nout, nin)
    b = b0 .+ bstd.*randn(nout)
    x = similar(b)
    out = similar(x)
    ∂W = similar(W)
    ∇b = similar(x)
    δ = similar(x, nin)
    Dense{Float64, F}(W, b, σ, x, out, ∂W, ∇b, δ)
end

""" out = l(z)
    Compute the output `out` from the layer.
    Store the input to the activation function in l.x and the output in l.out. """
function (l::Dense)(z)
    # Calculate intermediate layer v
    v = l.W * z + l.b

    # Cache result for back propagation
    l.x .= v

    # Calculate acitivation of intermediate layer v
    out = l.σ.(v)

    # Cache result of activation
    l.out .= out
end

# A network is just a sequence of layers
struct Network{T,N<:Layer{T}}
    layers::Vector{N}
end

""" out = n(z)
    Comute the result of applying each layer in a network to the previous output. """
function (n::Network)(z)
    for layer in n.layers
        z = layer(z)
    end
    return z
end

l1 = Dense(3, 2, relu) # 3 outputs, 2 inputs
l2 = Dense(1, 3, sigmoid) # 1 output, 3 inputs
n = Network([l1, l2]) # 1 output, 2 inputs

""" δ = backprop!(l::Dense, δnext, zin)
    Assuming that layer `l` has been called with `zin`,
    calculate the l.δ = ∂L/∂zᵢ given δᵢ₊₁ and zᵢ,
    and save l.∂W = ∂L/∂Wᵢ and l.∇b = (∂L/∂bᵢ)ᵀ """
function backprop!(l::Dense, δnext, zin)
    l.∇b .= δnext .* derivative.(l.σ, l.W * zin + l.b)
    l.∂W .= l.∇b * zin'
    l.δ .= l.W' * l.∇b
    return l.δ
end


""" backprop!(n::Network, input, ∂J∂y)
    Assuming that network `n` has been called with `input`, i.e `y=n(input)`
    backpropagate and save all gradients in the network,
    where ∂J∂y is the gradient (∂J/∂y)ᵀ. """

function backprop!(n::Network, input, ∂J∂y)
    layers = n.layers
    # To the last layer, δᵢ₊₁ is ∂J∂y
    δ = ∂J∂y
    # Iterate through layers, starting at the end
    for i in length(layers):-1:2
        δ = backprop!(layers[i], δ, layers[i-1].x)
    end
    # To first layer, the input was `input`
    zin = input
    δ = backprop!(layers[1], δ, zin)
    return
end


#Test that backprop works for NN
l1 = Dense(3, 3, sigmoid) #Layer with 3 input 3 outputs
l2 = Dense(1, 3, sigmoid) #Layer with 3 input 1 output
n = Network([l1,l2]) #NN with l1 and l2
z = [1. , 2. , 3.] #input to NN
out1 = copy(n(z)) #output from NN for z input

∂J∂W = [1.0] #Set loss to 1
backprop!(n, z, ∂J∂W) #Calculate all gradients using backprop

n.layers[1].∂W[1,1] #Should be same as this.

n.layers[1].W[1,1] += 0.0001 #Change element 1,1 in first layer with 0.0001

out2 = copy(n(z)) #output from changed NN for z input
(out2-out1)./0.0001 #Calculate difference in outputs scaled with diff



# This can be used to get a list of all parameters and gradients from a Dense layer
getparams(l::Dense) = ([l.W, l.b], [l.∂W, l.∇b])

""" `params, gradients = getparams(n::Network)`
    Return a list of references to all paramaters and corresponding gradients. """
function getparams(n::Network{T}) where T
    params = Array{T}[]         # List of references to vectors and matrices (arrays) of parameters
    gradients = Array{T}[]      # List of references to vectors and matrices (arrays) of gradients
    for layer in n.layers
        p, g = getparams(layer)
        append!(params, p)      # push the parameter references to params list
        append!(gradients, g)   # push the gradient references to gradients list
    end
    return params, gradients
end

### Define loss function L(y,yhat)
sumsquares(yhat,y) =  norm(yhat-y)^2
# And its gradient with respect to yhat: L_{yhat}(yhat,y)
derivative(::typeof(sumsquares), yhat, y) =  yhat - y


"""Function from assignment 3 manual"""
function gradientstep!(n, lossfunc, x, y)
    out = n(x)
    # Calculate (∂L/∂out)ᵀ
    ∇L = derivative(lossfunc, out, y)
    # Backward pass over network
    backprop!(n, x, ∇L)
    # Get list of all parameters and gradients
    parameters, gradients = getparams(n)
    # For each parameter, take gradient step
    for i = 1:length(parameters)
         p = parameters[i]
         g = gradients[i]
         # Update this parameter with a small step in negative gradient
         #→ direction
         p .= p .- 0.001.*g
         # The parameter p is either a W, or b so we broadcast to update all the
         #→ elements
    end
end

n = Network([Dense(3, 1, sigmoid), Dense(1, 3, sigmoid)])
x = randn(1)
y = [1.0] # We want the output to be 1

n(x) # This is probably not close to 1

gradientstep!(n, sumsquares, x, y)

n(x)

""" Structure for saving all the parameters and states needed for ADAM,
    as well as references to the parameters and gradients """
struct ADAMTrainer{T,GT}
    n::Network{T}
    β1::T
    β2::T
    ϵ::T
    γ::T
    params::GT              # List of paramaters in the network (all Wᵢ and bᵢ)
    gradients::GT           # List of gradients (all ∂Wᵢ and ∇bᵢ)
    ms::GT                  # List of mₜ for each parameter
    mhs::GT                 # List of \hat{m}ₜ for each parameter
    vs::GT                  # List of vₜ for each parameter
    vhs::GT                 # List of \hat{v}ₜ for each parameter
    t::Base.RefValue{Int}   # Reference to iteration counter
end

function ADAMTrainer(n::Network{T}, β1 = 0.9, β2 = 0.999, ϵ=1e-8, γ=0.1) where T
    params, gradients = getparams(n)
    ms = [zero(gi) for gi in gradients]
    mhs = [zero(gi) for gi in gradients]
    vs = [ones(size(gi)...) for gi in gradients]
    vhs = [zero(gi) for gi in gradients]
    ADAMTrainer{T, typeof(params)}(n, β1, β2, ϵ, γ, params, gradients, ms, mhs, vs, vhs, Ref(1))
end

""" `update!(At::ADAMTrainer)`
    Assuming that all gradients are already computed using backpropagation,
    take a step with the ADAM algorithm """
function update!(At::ADAMTrainer)
    # Get some of the variables that we need from the ADAMTrainer
    β1, β2, ϵ, γ = At.β1, At.β2, At.ϵ, At.γ
    # At.t is a reference, we get the value t like this
    t = At.t[]
    # For each of the W and b in the network
    for i in eachindex(At.params)
        p = At.params[i]        # This will reference either a W or b
        ∇p = At.gradients[i]    # This will reference either a ∂W or ∇b
        # Get each of the stored values m, mhat, v, vhat for this parameter
        m, mh, v, vh = At.ms[i], At.mhs[i], At.vs[i], At.vhs[i]

        m .= β1 .* m .+ (1 - β1) .* ∇p
        mh .= m ./ (1 - β1^t)
        v .= β2 .* v .+ (1 - β2) .* ∇p.^2
        vh .= v ./ (1 - β1^t)
        p .= p .- γ .* mh ./ (sqrt.(vh) .- ϵ)

    end
    At.t[] = t+1     # At.t is a reference, we update the value t like this
    return
end


""" `loss = train!(n, alg, xs, ys, lossfunc)`

    Train a network `n` with algorithm `alg` on inputs `xs`, expected outputs `ys`
    for loss-function `lossfunc` """
function train!(n, alg, xs, ys, lossfunc)
    lossall = 0.0           # This will keep track of the sum of the losses

    for i in eachindex(xs)  # For each data point
        xi = xs[i]          # Get data
        yi = ys[i]          # And expected output

        #Forward pass
        out = n(xi)

        #derivative of loss
        ∂J∂y = derivative(lossfunc, out, yi)

        #Backward pass
        backprop!(n, xi, ∂J∂y)

        update!(alg)

        loss = lossfunc(out, yi)
        lossall += loss
    end
    # Calculate and print avergae loss
    avgloss = lossall/length(xs)
    println("Avg loss: $avgloss")
    return avgloss
end

""" `testloss(n, xs, ys, lossfunc)`
    Evaluate mean loss of network `n`, over data `xs`, `ys`,
    using lossfunction `lossfunc` """
getloss(n, xs, ys, lossfunc) = mean(xy -> lossfunc(xy[2], n(xy[1])), zip(xs,ys))


#########################################################
#########################################################
#########################################################
### Task 3:

### Define network
# We use some reasonable value on initial weights
l1 = Dense(30, 1, leakyrelu, 0.0, 3.0, 0.0, 0.1)
lis = [Dense(30, 30, leakyrelu, 0.0, 3.0, 0.0, 0.1) for i = 1:4]
# Last layer has no activation function (identity)
ln = Dense(1, 30, identity, 0.0, 1.0, 0.0, 0.1)
n = Network([l1, lis..., ln])

### This is the function we want to approximate
fsol(x) = [min(3,norm(x)^2)]

### Define data, in range [-4,4]
xs = [rand(1).*8 .- 4 for i = 1:2000]
ys = [fsol(xi) for xi in xs]
# Test data
testxs = [rand(1).*8 .- 4 for i = 1:1000]
testys = [fsol(xi) for xi in testxs]

### Define algorithm
adam = ADAMTrainer(n, 0.95, 0.999, 1e-8, 0.0001)

### Train and plot
using Plots
# Train once over the data set
@time train!(n, adam, xs, ys, sumsquares)
scatter(xs, [copy(n(xi)) for xi in xs])

# Train 100 times over the data set
for i = 1:100
    # Random ordering of all the data
    Iperm = randperm(length(xs))
    @time train!(n, adam, xs[Iperm], ys[Iperm], sumsquares)
end

# Plot real line and prediction
plot(-4:0.01:4, [fsol.(xi)[1] for xi in -4:0.01:4], c=:blue)
scatter!(xs, ys, lab="", m=(:cross,0.2,:blue))
scatter!(xs, [copy(n(xi)) for xi in xs], m=(:circle,0.2,:red))

# We can calculate the mean error over the training data like this also
getloss(n, xs, ys, sumsquares)
# Loss over test data like this
getloss(n, testxs, testys, sumsquares)

# Plot expected line
plot(-8:0.01:8, [fsol.(xi)[1] for xi in -8:0.01:8], c=:blue);
# Plot full network result
plot!(-8:0.01:8, [copy(n([xi]))[1] for xi in -8:0.01:8], c=:red)

#########################################################
#########################################################
#########################################################
### Task 4:

getloss(n, xs, ys, sumsquares)
getloss(n, testxs, testys, sumsquares)
#########################################################
#########################################################
#########################################################
### Task 5:


getloss(n, xs, ys, sumsquares)
getloss(n, testxs, testys, sumsquares)
#########################################################
#########################################################
#########################################################
### Task 6:
fsol(x) = [min(0.5,sin(0.5*norm(x)^2))]

getloss(n, xs, ys, sumsquares)
getloss(n, testxs, testys, sumsquares)

# Plotttnig that can be used for task 6:
scatter3d([xi[1] for xi in xs], [xi[2] for xi in xs], [n(xi)[1] for xi in xs], m=(:blue,1, :cross, stroke(0, 0.2, :blue)), size=(1200,800));
scatter3d!([xi[1] for xi in xs], [xi[2] for xi in xs], [yi[1] for yi in ys], m=(:red,1, :circle, stroke(0, 0.2, :red)), size=(1200,800))
