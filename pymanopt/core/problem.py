"""The Pymanopt problem class."""

import functools
from typing import Callable, Optional

import numpy as np

from ..autodiff import Function
from ..manifolds.manifold import Manifold


class Problem:
    """Problem class to define a Riemannian optimization problem.

    Args:
        manifold: Manifold to optimize over.
        cost: A callable decorated with a decorator from
            :mod:`pymanopt.functions` which takes a point on a manifold and
            returns a real scalar.
            If any decorator other than
            :func:`pymanopt.function.Callable` is used the gradient and
            Hessian-vector production functions are computed automatically if
            needed and no ``(e)grad`` or ``(e)hess`` arguments are provided.
        egrad: The Euclidean gradient, i.e., the gradient of the cost function
            in the typical sense in the ambient space.
            The returned value need not belong to the tangent space of
            ``manifold``.
        ehess: The Euclidean Hessian-vector product, i.e., the directional
            derivative of ``egrad`` in the direction of a tangent vector.
        grad: The Riemannian gradient.
            For embeddes submanifolds this is simply the projection of
            ``egrad`` on the tangent space of ``manifold``.
            In most cases this need not be provided and the Riemannian gradient
            is instead computed internally.
            If provided, the function needs to return a vector in the tangent
            space of ``manifold``.
        hess: The Riemannian Hessian-vector product, i.e., the directional
            derivative of ``grad`` in the direction of a tangent vector.
            As with ``grad`` this usually need not be provided explicitly.
        verbosity: Level of information printed by the solver while it
            operates: 0 is silent, 2 is most verbose.
    """

    def __init__(
        self,
        manifold: Manifold,
        cost: Function,
        egrad: Optional[Function] = None,
        ehess: Optional[Function] = None,
        grad: Optional[Function] = None,
        hess: Optional[Function] = None,
        precon: Optional[Callable] = None,
        verbosity: int = 2,
    ):
        self.manifold = manifold

        for function, name in (
            (cost, "cost"),
            (egrad, "egrad"),
            (ehess, "ehess"),
            (grad, "grad"),
            (hess, "hess"),
        ):
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
            if key == "verbosity" and (
                not isinstance(value, int) or value < 0
            ):
                raise ValueError(
                    "Verbosity level must be an nonnegative integer"
                )
            if key in ("manifold", "precon"):
                raise AttributeError(f"Cannot override '{key}' attribute")
        super().__setattr__(key, value)

    @staticmethod
    def _validate_function(function, name):
        if function is not None and not isinstance(function, Function):
            raise ValueError(
                f"Function '{name}' must be decorated with a backend decorator"
            )

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
        """Decorator to group return values according to a given signature.

        Wraps a function inside another function which groups the return
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
                    "Function returned an unexpected number of arguments"
                )
            groups = []
            i = 0
            for group_size in signature:
                if group_size == 1:
                    group = return_values[i]
                else:
                    group = return_values[i : i + group_size]
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
                    *self._flatten_arguments(vector, point_layout),
                )

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
                self._original_cost.compute_gradient()
            )
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
                self._original_cost.compute_hessian_vector_product()
            )
        return self._ehess

    @property
    def hess(self):
        if self._hess is None:
            ehess = self.ehess

            def hess(x, a):
                return self.manifold.ehess2rhess(
                    x, self.egrad(x), ehess(x, a), a
                )

            self._hess = hess
        return self._hess
