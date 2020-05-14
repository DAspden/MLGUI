

#using Pkg
#Pkg.add("Flux")
using Flux

using Flux.Tracker

# executable math
f(x) = x^2+1

# f'(x) = 2x
df(x) = gradient(f,x,nest=true)[1] # df is a tuple, [1] gets the first coordinate


df(4)



# f''(x) = 2
ddf(x) = gradient(df,x,nest=true)[1]
ddf(0)



h(x) = -cos(x)^cos(x)

# h'(x) = tan(x)cos(x)^(cos(x)+1)(log(cos(x))+1) obviously
dh(x)=gradient(h,x)[1]
dh(pi/4)

f(x,y,z) = x^2 + y^2 + z^2

#grad(f) = (2x,2y,2z)
gradient(f,1,2,3)


# Quick Example to introduce Params(): Linear Regression

# random initial parameters
W = rand(5,10)
b = rand(5)

fhat(x) = W*x + b

function loss(x,y)
    yhat = fhat(x) # our prediction for y
    return sum((y-yhat).^2)
end

x = rand(10)
y = rand(5)

loss(x,y) # big loss with random parameters
# I have 50+ paramters, how do I pass them all at once?

W = param(W)
b = param(b)

grads = gradient(() -> loss(x, y), Params([W, b]))


# gradient descent

alpha = 0.01 # learning rate, step size, etc
gW = grads[W]
gb = grads[b]

Tracker.update!(W,-alpha*gW) # essentially W = W - alpha * gW. It does something else I don't understand
Tracker.update!(b,-alpha*gb);

loss(x,y)



function sigmoid(x)
    return 1/(1+exp(-x))
end

W1 = param(rand(7, 10))
b1 = param(rand(7))
layer1(x) = W1 * x .+ b1

W2 = param(rand(5, 7))
b2 = param(rand(5))
layer2(x) = W2 * x .+ b2

model(x) = layer2(sigmoid.(layer1(x)))



layer1(x) = Dense(10,7,sigmoid)
layer2(x) = Dense(7,5)

model(x) = layer2(layer1(x))

# or equivalently
model2(x) = Chain(layer1,layer2)

# cool thing about Chain is that it supports indexing
model2(x)[1]



# To train a model call somthing like


train!(objective, parameters, data, optimizer, cb = () -> println("still training..."))

# cb stands for callback. Its useful to updating you about training (e.g. what the loss is currently)
# By default, it is called after every batch. Use Flux.throttle() to change this

@epoch 5 train!(...)
