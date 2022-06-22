import functools
from typing import Sequence

import numpy as np

from pymanopt.manifolds.manifold import Manifold
from pymanopt.tools import ndarraySequenceMixin, return_as_class_instance


class Product(Manifold):
    """Cartesian product manifold.

    Points on the manifold and tangent vectors are represented as lists of
    points and tangent vectors of the individual manifolds.
    The metric is obtained by element-wise extension of the individual
    manifolds.

    Args:
        manifolds: The collection of manifolds in the product.
    """

    def __init__(self, manifolds: Sequence[Manifold]):
        for manifold in manifolds:
            if isinstance(manifold, Product):
                raise ValueError("Nested product manifolds are not supported")
        self.manifolds = tuple(manifolds)
        manifold_names = " x ".join([str(manifold) for manifold in manifolds])
        name = f"Product manifold: {manifold_names}"

        dimension = np.sum([manifold.dim for manifold in manifolds])
        point_layout = tuple(manifold.point_layout for manifold in manifolds)
        super().__init__(name, dimension, point_layout=point_layout)

    @property
    def typical_dist(self):
        return np.sqrt(
            np.sum([manifold.typical_dist**2 for manifold in self.manifolds])
        )

    def _dispatch(
        self,
        method_name,
        *,
        transform=lambda value: value,
        reduction=lambda values: values,
    ):
        """Wrapper to delegate method calls to individual manifolds."""

        @functools.wraps(getattr(self, method_name))
        def wrapper(*args, **kwargs):
            return_values = [
                transform(getattr(manifold, method_name)(*arguments))
                for manifold, *arguments in zip(self.manifolds, *args)
            ]
            return reduction(return_values)

        return wrapper

    def norm(self, point, tangent_vector):
        return np.sqrt(
            self.inner_product(point, tangent_vector, tangent_vector)
        )

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return self._dispatch("inner_product", reduction=np.sum)(
            point, tangent_vector_a, tangent_vector_b
        )

    def dist(self, point_a, point_b):
        return self._dispatch(
            "dist",
            transform=lambda value: value**2,
            reduction=lambda values: np.sqrt(np.sum(values)),
        )(point_a, point_b)

    def projection(self, point, vector):
        return self._dispatch("projection", reduction=_ProductTangentVector)(
            point, vector
        )

    def to_tangent_space(self, point, vector):
        return self._dispatch(
            "to_tangent_space", reduction=_ProductTangentVector
        )(point, vector)

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return self._dispatch(
            "euclidean_to_riemannian_gradient", reduction=_ProductTangentVector
        )(point, euclidean_gradient)

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        return self._dispatch(
            "euclidean_to_riemannian_hessian", reduction=_ProductTangentVector
        )(point, euclidean_gradient, euclidean_hessian, tangent_vector)

    def exp(self, point, tangent_vector):
        return self._dispatch("exp")(point, tangent_vector)

    def retraction(self, point, tangent_vector):
        return self._dispatch("retraction")(point, tangent_vector)

    def log(self, point_a, point_b):
        return self._dispatch("log", reduction=_ProductTangentVector)(
            point_a, point_b
        )

    def random_point(self):
        return self._dispatch("random_point")()

    def random_tangent_vector(self, point):
        scale = len(self.manifolds) ** (-1 / 2)
        return self._dispatch(
            "random_tangent_vector",
            transform=lambda value: scale * value,
            reduction=_ProductTangentVector,
        )(point)

    def transport(self, point_a, point_b, tangent_vector_a):
        return self._dispatch("transport", reduction=_ProductTangentVector)(
            point_a, point_b, tangent_vector_a
        )

    def pair_mean(self, point_a, point_b):
        return self._dispatch("pair_mean")(point_a, point_b)

    def zero_vector(self, point):
        return self._dispatch("zero_vector", reduction=_ProductTangentVector)(
            point
        )


class _ProductTangentVector(ndarraySequenceMixin, list):
    @return_as_class_instance(unpack=False)
    def __add__(self, other):
        if len(self) != len(other):
            raise ValueError("Arguments must be same length")
        return [v + other[k] for k, v in enumerate(self)]

    @return_as_class_instance(unpack=False)
    def __sub__(self, other):
        if len(self) != len(other):
            raise ValueError("Arguments must be same length")
        return [v - other[k] for k, v in enumerate(self)]

    @return_as_class_instance(unpack=False)
    def __mul__(self, other):
        return [other * val for val in self]

    __rmul__ = __mul__

    @return_as_class_instance(unpack=False)
    def __truediv__(self, other):
        return [val / other for val in self]

    @return_as_class_instance(unpack=False)
    def __neg__(self):
        return [-val for val in self]
