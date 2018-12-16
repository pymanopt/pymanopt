"""
Module containing functions to compile and differentiate Theano graphs. Part of
the pymanopt package.

Jamie Townsend December 2014
"""
import itertools

try:
    import theano
    import theano.tensor as T
    from theano.gradient import disconnected_grad
except ImportError:
    theano = None
    T = None

from ._backend import Backend, assert_backend_available
from .. import make_function_decorator_with_argument
from ...tools import unpack_arguments, flatten_args, group_return_values


class TheanoBackend(Backend):
    def __str__(self):
        return "theano"

    @staticmethod
    def is_available():
        return theano is not None

    @assert_backend_available
    def is_compatible(self, objective, argument):
        args = flatten_args(argument)
        return all([isinstance(arg, T.TensorVariable) for arg in args])

    @assert_backend_available
    def compile_function(self, objective, argument):
        """
        Wrapper for theano.function(). Compiles a theano graph into a python
        function.
        """
        args = flatten_args(argument)
        func = theano.function(inputs=args, outputs=objective)
        if len(args) == 1:
            return func
        return unpack_arguments(func)

    @assert_backend_available
    def compute_gradient(self, objective, argument):
        """
        Returns a compiled function computing the gradient of 'objective' with
        respect to 'argument'.
        """
        args = flatten_args(argument)
        # For cost functions expecting one argument, make sure we don't return
        # the gradient evaluated at some point as a singleton tuple.
        if len(args) == 1:
            (arg,) = args
            g = T.grad(objective, arg)
            return theano.function(args, g)

        g = T.grad(objective, args)
        grad = theano.function(args, g)
        return group_return_values(unpack_arguments(grad), argument)

    def _compute_unary_hvp(self, g, argument):
        """
        Returns a function accepting two arguments to compute a Hessian-vector
        product of a scalar-valued unary function.
        """
        u = argument.type()
        try:
            R = T.Rop(g, argument, u)
        except NotImplementedError:
            proj = T.sum(g * disconnected_grad(u))
            R = T.grad(proj, argument)
        return theano.function([argument, u], R, on_unused_input="warn")

    def _compute_nary_hvp(self, g, args):
        """
        Returns a function accepting 2 * len(args) arguments to compute a
        Hessian-vector product of a multivariate function defined on a product
        manifold.
        """
        u = [arg.type() for arg in args]
        try:
            R = T.Rop(g, args, u)
        except NotImplementedError:
            # TODO(nkoep): Write a test case for this path.
            # Implementation based on:
            #   tensorflow.python.ops.gradients_impl._hessian_vector_product
            proj = [T.sum(gi * disconnected_grad(ui)) for gi, ui in zip(g, u)]
            proj_grad = [T.grad(proj_elem, args,
                                disconnected_inputs="ignore",
                                return_disconnected="None")
                         for proj_elem in proj]
            proj_grad_transpose = map(list, zip(*proj_grad))
            proj_grad_stack = [
                T.stacklists([c for c in row if c is not None])
                for row in proj_grad_transpose]
            R = [T.sum(stack, axis=0) for stack in proj_grad_stack]
        return theano.function(list(itertools.chain(args, u)), R,
                               on_unused_input="warn")

    @assert_backend_available
    def compute_hessian(self, objective, argument):
        """
        Computes the directional derivative of the gradient (which is equal to
        the Hessian multiplied by direction).
        """
        # See above.
        args = flatten_args(argument)
        if len(args) == 1:
            (arg,) = args
            g = T.grad(objective, arg)
            return self._compute_unary_hvp(g, arg)

        g = T.grad(objective, args)
        hess = self._compute_nary_hvp(g, args)
        def wrapper(x, u):
            return hess(*itertools.chain(flatten_args(x), flatten_args(u)))
        return group_return_values(wrapper, argument)


Theano = make_function_decorator_with_argument(TheanoBackend)
