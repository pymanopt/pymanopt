"""
Module containing pymanopt problem class. Use this to build a problem
object to feed to one of the solvers.
"""
import functools

import numpy as np

from ..autodiff import Function


class Problem:
    """
    Problem class for setting up a problem to feed to one of the
    pymanopt solvers.

    Attributes:
        - manifold
            Manifold to optimize over.
        - cost
            A callable which takes an element of manifold and returns a
            real number, or a symbolic Theano or TensorFlow expression.
            In case of a symbolic expression, the gradient (and if
            necessary the Hessian) are computed automatically if they are
            not explicitly given. We recommend you take this approach
            rather than calculating gradients and Hessians by hand.
        - grad
            grad(x) is the gradient of cost at x. This must take an
            element X of manifold and return an element of the tangent space
            to manifold at X. This is usually computed automatically and
            doesn't need to be set by the user.
        - hess
            hess(x, a) is the directional derivative of grad at x, in
            direction a. It should return an element of the tangent
            space to manifold at x.
        - egrad
            The 'Euclidean gradient', egrad(x) should return the grad of
            cost in the usual sense, i.e. egrad(x) need not lie in the
            tangent space.
        - ehess
            The 'Euclidean Hessian', ehess(x, a) should return the
            directional derivative of egrad at x in direction a. This
            need not lie in the tangent space.
        - verbosity (2)
            Level of information printed by the solver while it operates, 0
            is silent, 2 is most information.
    """

    def __init__(self, manifold, cost, egrad=None, ehess=None, grad=None,
                 hess=None, precon=None, verbosity=2):
        self.manifold = manifold

        for function, name in (
                (cost, "cost"), (egrad, "egrad"), (ehess, "ehess"),
                (grad, "grad"), (hess, "hess")):
            self._validate_function(function, name)

        self._original_cost = cost
        self._cost = self._wrap_function(cost)

        if egrad is not None:
            egrad = self._wrap_gradient(egrad)
        self._egrad = egrad
        if ehess is not None:
            ehess = self._wrap_hessian_vector_product(ehess)
        self._ehess = ehess

        if grad is not None:
            grad = self._wrap_gradient(grad)
        self._grad = grad
        if hess is not None:
            hess = self._wrap_hessian_vector_product(hess)
        self._hess = hess

        if precon is None:
            def precon(x, d):
                return d
        self.precon = precon

        self.verbosity = verbosity

    def __setattr__(self, key, value):
        if hasattr(self, key):
            if (key == "verbosity"
                    and (not isinstance(value, int) or value < 0)):
                raise ValueError(
                    "Verbosity level must be an nonnegative integer")
            if key in ("manifold", "precon"):
                raise AttributeError(
                    "Cannot override '{:s}' attribute".format(key))
        super().__setattr__(key, value)

    @staticmethod
    def _validate_function(function, name):
        if function is not None and not isinstance(function, Function):
            raise ValueError(
                "Function '{:s}' must be decorated with one of the decorators "
                "from 'pymanopt.function'".format(name))

    def _flatten_arguments(self, arguments, signature):
        assert len(arguments) == len(signature)

        flattened_arguments = []
        for i, group_size in enumerate(signature):
            argument = arguments[i]
            if group_size == 1:
                assert not isinstance(argument, (list, tuple))
                flattened_arguments.append(argument)
            else:
                assert len(argument) == group_size
                flattened_arguments.extend(argument)
        return flattened_arguments

    def _group_return_values(self, function, signature):
        """Wraps a function inside another function which groups the return
        values of ``function`` according to the group sizes delineated by
        ``signature``.
        """
        assert all((isinstance(group, int) for group in signature))

        num_return_values = np.sum(signature)

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return_values = function(*args, **kwargs)
            if not isinstance(return_values, (list, tuple)):
                raise ValueError("Function returned an unexpected value")
            if len(return_values) != num_return_values:
                raise ValueError(
                    "Function returned an unexpected number of arguments")
            groups = []
            i = 0
            for group_size in signature:
                if group_size == 1:
                    group = return_values[i]
                else:
                    group = return_values[i:i+group_size]
                groups.append(group)
                i += group_size
            return groups
        return wrapper

    def _wrap_function(self, function):
        point_layout = self.manifold.point_layout
        if isinstance(point_layout, (tuple, list)):
            @functools.wraps(function)
            def wrapper(point):
                return function(*self._flatten_arguments(point, point_layout))
            return wrapper

        assert isinstance(point_layout, int)

        if point_layout == 1:
            @functools.wraps(function)
            def wrapper(point):
                return function(point)
        else:
            @functools.wraps(function)
            def wrapper(point):
                return function(*point)
        return wrapper

    def _wrap_gradient(self, gradient):
        wrapped_gradient = self._wrap_function(gradient)
        point_layout = self.manifold.point_layout
        if isinstance(point_layout, (list, tuple)):
            return self._group_return_values(wrapped_gradient, point_layout)
        return wrapped_gradient

    def _wrap_hessian_vector_product(self, hessian_vector_product):
        point_layout = self.manifold.point_layout
        if isinstance(point_layout, (list, tuple)):
            @functools.wraps(hessian_vector_product)
            def wrapper(point, vector):
                return hessian_vector_product(
                    *self._flatten_arguments(point, point_layout),
                    *self._flatten_arguments(vector, point_layout))
            return self._group_return_values(wrapper, point_layout)

        if point_layout == 1:
            @functools.wraps(hessian_vector_product)
            def wrapper(point, vector):
                return hessian_vector_product(point, vector)
        else:
            @functools.wraps(hessian_vector_product)
            def wrapper(point, vector):
                return hessian_vector_product(*point, *vector)
        return wrapper

    @property
    def cost(self):
        return self._cost

    @property
    def egrad(self):
        if self._egrad is None:
            self._egrad = self._wrap_gradient(
                self._original_cost.compute_gradient())
        return self._egrad

    @property
    def grad(self):
        if self._grad is None:
            egrad = self.egrad

            def grad(x):
                return self.manifold.egrad2rgrad(x, egrad(x))
            self._grad = grad
        return self._grad

    @property
    def ehess(self):
        if self._ehess is None:
            self._ehess = self._wrap_hessian_vector_product(
                self._original_cost.compute_hessian_vector_product())
        return self._ehess

    @property
    def hess(self):
        if self._hess is None:
            ehess = self.ehess

            def hess(x, a):
                return self.manifold.ehess2rhess(
                    x, self.egrad(x), ehess(x, a), a)
            self._hess = hess
        return self._hess
