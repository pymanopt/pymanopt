"""
Module containing functions to compile and differentiate Theano graphs. Part of
the pymanopt package.

Jamie Townsend December 2014
"""
import theano.tensor as T
import theano


def compile(objective, argument):
    """
    Wrapper for the theano.function(). Compiles a theano graph into a python
    function.
    """
    return theano.function([argument], objective)

def gradient(objective, argument):
    """
    Wrapper for theano.tensor.grad().
    Compute the gradient of 'objective' with respect to 'argument' and return
    compiled version.
    """
    g = T.grad(objective, argument)
    return compile(g, argument)

def grad_hess(objective, argument):
    """
    Compute both the gradient and the directional derivative of the gradient
    (which is equal to the hessian multiplied by direction).
    """
    # TODO: Check that the hessian calculation is correct!
    # TODO: Make this compatible with non-matrix manifolds.
    g = T.grad(objective, argument)
    grad = compile(g, argument)

    # For now, this function will only work for matrix manifolds.
    A = T.matrix()
    n, p =  T.shape(argument)
    try:
        # First attempt efficient 'R-op', this directly calculates the
        # directional derivative of the gradient, rather than explicitly
        # calculating the hessian and then multiplying.
        R = T.Rop(g, argument, A)
    except NotImplementedError:
        H = T.jacobian(g.flatten(), argument).reshape([n,p,n,p],4)
        R = T.tensordot(H, A)

    hess = theano.function([argument, A], R)
    return grad, hess
