"""
Module containing functions to compile and differentiate Theano graphs. Part of
the pymanopt package.

Jamie Townsend December 2014
"""
import theano
import theano.tensor as T

from warnings import warn

from ._backend import Backend


class TheanoBackend(Backend):
    def compile_function(self, objective, argument):
        """
        Wrapper for the theano.function(). Compiles a theano graph into a
        python function.
        """
        return theano.function([argument], objective)

    def compute_gradient(self, objective, argument):
        """
        Wrapper for theano.tensor.grad(). Computes the gradient of 'objective'
        with respect to 'argument' and returns compiled version.
        """
        g = T.grad(objective, argument)
        return self.compile_function(g, argument)

    def compute_hessian(self, objective, argument):
        """
        Computes the directional derivative of the gradient (which is equal to
        the Hessian multiplied by direction).
        """
        g = T.grad(objective, argument)

        # Create a new tensor A, which has the same type (i.e. same
        # dimensionality) as argument.
        A = argument.type()

        try:
            # First attempt efficient 'R-op', this directly calculates the
            # directional derivative of the gradient, rather than explicitly
            # calculating the Hessian and then multiplying.
            R = T.Rop(g, argument, A)
        except NotImplementedError:
            shp = T.shape(argument)
            H = T.jacobian(g.flatten(), argument).reshape(
                T.concatenate([shp, shp]), 2 * A.ndim)
            R = T.tensordot(H, A, A.ndim)

        try:
            hess = theano.function([argument, A], R, on_unused_input="raise")
        except theano.compile.UnusedInputError:
            warn("Theano detected unused input - suggests Hessian may be zero "
                 "or constant.")
            hess = theano.function([argument, A], R, on_unused_input="ignore")
        return hess
