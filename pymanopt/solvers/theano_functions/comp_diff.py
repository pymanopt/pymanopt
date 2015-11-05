# Module containing functions to compile and differentiate Theano graphs.

import theano.tensor as T
import theano

# Compile objective function defined in Theano.
def compile(objective, argument):
    return theano.function([argument], objective)

# Compute the gradient of 'objective' with respect to 'argument' and return
# compiled function.
def gradient(objective, argument):
    g = T.grad(objective, argument)
    return theano.function([argument], g)