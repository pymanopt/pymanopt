"""
Module containing functions to compile and differentiate Theano graphs. Part of
the pymanopt package.

Jamie Townsend December 2014
"""
try:
    import theano
    import theano.tensor as T
except ImportError:
    theano = None
    T = None

from ._backend import Backend, assert_backend_available


class TheanoBackend(Backend):
    def __str__(self):
        return "theano"

    def is_available(self):
        return theano is not None and T is not None

    @assert_backend_available
    def is_compatible(self, objective, argument):
        if isinstance(objective, T.TensorVariable):
            if (argument is None or not
                isinstance(argument, T.TensorVariable) and not
                all([isinstance(arg, T.TensorVariable)
                     for arg in argument])):
                raise ValueError(
                    "Theano backend requires an argument (or sequence of "
                    "arguments) with respect to which compilation is to be "
                    "carried out")
            return True
        return False

    @assert_backend_available
    def compile_function(self, objective, argument):
        """
        Wrapper for the theano.function(). Compiles a theano graph into a
        python function.
        """
        try:
            return theano.function([argument], objective)
        except TypeError:
            # Assume we are on a product manifold
            compiled = theano.function([arg for arg in argument], objective)
            return lambda x: compiled(*x)

    @assert_backend_available
    def compute_gradient(self, objective, argument):
        """
        Wrapper for theano.tensor.grad(). Computes the gradient of 'objective'
        with respect to 'argument' and returns compiled version.
        """
        g = T.grad(objective, argument)
        return self.compile_function(g, argument)

    @assert_backend_available
    def compute_hessian(self, objective, argument):
        """
        Computes the directional derivative of the gradient (which is equal to
        the Hessian multiplied by direction).
        """
        g = T.grad(objective, argument)

        # Create a new tensor A, which has the same type (i.e. same
        # dimensionality) as argument.
        try:
            A = argument.type()
        except AttributeError:
            # Assume we are on the product manifold
            A = [arg.type() for arg in argument]

        try:
            # First attempt efficient 'R-op', this directly calculates the
            # directional derivative of the gradient, rather than explicitly
            # calculating the Hessian and then multiplying.
            R = T.Rop(g, argument, A)
        except NotImplementedError:
            # TODO: fix this fallback for the product manifold.
            shp = T.shape(argument)
            H = T.jacobian(g.flatten(), argument).reshape(
                T.concatenate([shp, shp]), 2 * A.ndim)
            R = T.tensordot(H, A, A.ndim)

        try:
            hess = theano.function([argument, A], R, on_unused_input="warn")
        except TypeError:
            hess_prod = theano.function(argument + A, R,
                                        on_unused_input="warn")

            def hess(x, a):
                return hess_prod(*(x + a))

        return hess
