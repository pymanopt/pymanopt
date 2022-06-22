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
            If any decorator other than :func:`pymanopt.function.numpy` is
            used, the gradient and Hessian functions are generated
            automatically if needed and no ``{euclidean,riemannian}_gradient``
            or ``{euclidean,riemannian}_hessian`` arguments are provided.
        euclidean_gradient: The Euclidean gradient, i.e., the gradient of the
            cost function in the typical sense in the ambient space.
            The returned value need not belong to the tangent space of
            ``manifold``.
        riemannian_gradient: The Riemannian gradient.
            For embedded submanifolds this is simply the projection of
            ``euclidean_gradient`` on the tangent space of ``manifold``.
            In most cases this need not be provided and the Riemannian gradient
            is instead computed internally.
            If provided, the function needs to return a vector in the tangent
            space of ``manifold``.
        euclidean_hessian: The Euclidean Hessian, i.e., the directional
            derivative of ``euclidean_gradient`` in the direction of a tangent
            vector.
        riemannian_hessian: The Riemannian Hessian, i.e., the directional
            derivative of ``riemannian_gradient`` in the direction of a tangent
            vector.
            As with ``riemannian_gradient`` this usually need not be provided
            explicitly.
    """

    def __init__(
        self,
        manifold: Manifold,
        cost: Function,
        *,
        euclidean_gradient: Optional[Function] = None,
        riemannian_gradient: Optional[Function] = None,
        euclidean_hessian: Optional[Function] = None,
        riemannian_hessian: Optional[Function] = None,
        preconditioner: Optional[Callable] = None,
    ):
        self.manifold = manifold

        for function, name in (
            (cost, "cost"),
            (euclidean_gradient, "euclidean_gradient"),
            (euclidean_hessian, "euclidean_hessian"),
            (riemannian_gradient, "riemannian_gradient"),
            (riemannian_hessian, "riemannian_hessian"),
        ):
            self._validate_function(function, name)

        if euclidean_gradient is not None and riemannian_gradient is not None:
            raise ValueError(
                "Only 'euclidean_gradient' or 'riemannian_gradient' should be "
                "provided, not both"
            )
        if euclidean_hessian is not None and riemannian_hessian is not None:
            raise ValueError(
                "Only 'euclidean_hessian' or 'riemannian_hessian' should be "
                "provided, not both"
            )

        self._original_cost = cost
        self._cost = self._wrap_function(cost)

        if euclidean_gradient is not None:
            euclidean_gradient = self._wrap_gradient_operator(
                euclidean_gradient
            )
        self._euclidean_gradient = euclidean_gradient
        if euclidean_hessian is not None:
            euclidean_hessian = self._wrap_hessian_operator(
                euclidean_hessian, embed_tangent_vectors=True
            )
        self._euclidean_hessian = euclidean_hessian

        if riemannian_gradient is not None:
            riemannian_gradient = self._wrap_gradient_operator(
                riemannian_gradient
            )
        self._riemannian_gradient = riemannian_gradient
        if riemannian_hessian is not None:
            riemannian_hessian = self._wrap_hessian_operator(
                riemannian_hessian
            )
        self._riemannian_hessian = riemannian_hessian

        if preconditioner is None:

            def preconditioner(point, tangent_vector):
                return tangent_vector

        self.preconditioner = preconditioner

    def __setattr__(self, key, value):
        if hasattr(self, key) and key in ("manifold", "preconditioner"):
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

    def _wrap_gradient_operator(self, gradient):
        wrapped_gradient = self._wrap_function(gradient)
        point_layout = self.manifold.point_layout
        if isinstance(point_layout, (list, tuple)):
            return self._group_return_values(wrapped_gradient, point_layout)
        return wrapped_gradient

    def _wrap_hessian_operator(
        self, hessian_operator, *, embed_tangent_vectors=False
    ):
        point_layout = self.manifold.point_layout
        if isinstance(point_layout, (list, tuple)):

            @functools.wraps(hessian_operator)
            def wrapper(point, vector):
                return hessian_operator(
                    *self._flatten_arguments(point, point_layout),
                    *self._flatten_arguments(vector, point_layout),
                )

            wrapper = self._group_return_values(wrapper, point_layout)

        elif point_layout == 1:

            @functools.wraps(hessian_operator)
            def wrapper(point, vector):
                return hessian_operator(point, vector)

        else:

            @functools.wraps(hessian_operator)
            def wrapper(point, vector):
                return hessian_operator(*point, *vector)

        if embed_tangent_vectors:

            @functools.wraps(wrapper)
            def hvp(point, vector):
                return wrapper(point, self.manifold.embedding(point, vector))

        else:
            hvp = wrapper
        return hvp

    @property
    def cost(self):
        return self._cost

    @property
    def euclidean_gradient(self):
        if self._euclidean_gradient is None:
            self._euclidean_gradient = self._wrap_gradient_operator(
                self._original_cost.get_gradient_operator()
            )
        return self._euclidean_gradient

    @property
    def riemannian_gradient(self):
        if self._riemannian_gradient is None:

            def riemannian_gradient(point):
                return self.manifold.euclidean_to_riemannian_gradient(
                    point, self.euclidean_gradient(point)
                )

            self._riemannian_gradient = riemannian_gradient
        return self._riemannian_gradient

    @property
    def euclidean_hessian(self):
        if self._euclidean_hessian is None:
            self._euclidean_hessian = self._wrap_hessian_operator(
                self._original_cost.get_hessian_operator(),
                embed_tangent_vectors=True,
            )
        return self._euclidean_hessian

    @property
    def riemannian_hessian(self):
        if self._riemannian_hessian is None:

            def riemannian_hessian(point, tangent_vector):
                return self.manifold.euclidean_to_riemannian_hessian(
                    point,
                    self.euclidean_gradient(point),
                    self.euclidean_hessian(point, tangent_vector),
                    tangent_vector,
                )

            self._riemannian_hessian = riemannian_hessian
        return self._riemannian_hessian
