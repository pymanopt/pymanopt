import functools

import numpy as np

from pymanopt.manifolds.manifold import Manifold
from pymanopt.tools import ndarraySequenceMixin, return_as_class_instance


class Product(Manifold):
    """Product manifold, i.e., the cartesian product of multiple manifolds."""

    def __init__(self, manifolds):
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
        return np.sqrt(self.inner(point, tangent_vector, tangent_vector))

    def inner(self, point, tangent_vector_a, tangent_vector_b):
        return self._dispatch("inner", reduction=np.sum)(
            point, tangent_vector_a, tangent_vector_b
        )

    def dist(self, point_a, point_b):
        return self._dispatch(
            "dist",
            transform=lambda value: value**2,
            reduction=lambda values: np.sqrt(np.sum(values)),
        )(point_a, point_b)

    def projection(self, point, vector):
        return self._dispatch("proj", reduction=_ProductTangentVector)(
            point, vector
        )

    def egrad2rgrad(self, point, euclidean_gradient):
        return self._dispatch("egrad2rgrad", reduction=_ProductTangentVector)(
            point, euclidean_gradient
        )

    def ehess2rhess(
        self, point, euclidean_gradient, euclidean_hvp, tangent_vector
    ):
        return self._dispatch("ehess2rhess", reduction=_ProductTangentVector)(
            point, euclidean_gradient, euclidean_hvp, tangent_vector
        )

    def exp(self, point, tangent_vector):
        return self._dispatch("exp")(point, tangent_vector)

    def retraction(self, point, tangent_vector):
        return self._dispatch("retr")(point, tangent_vector)

    def log(self, point_a, point_b):
        return self._dispatch("log", reduction=_ProductTangentVector)(
            point_a, point_b
        )

    def rand(self):
        return self._dispatch("rand")()

    def randvec(self, point):
        scale = len(self.manifolds) ** (-1 / 2)
        return self._dispatch(
            "randvec",
            transform=lambda value: scale * value,
            reduction=_ProductTangentVector,
        )(point)

    def transport(self, point_a, point_b, tangent_vector_a):
        return self._dispatch("transp", reduction=_ProductTangentVector)(
            point_a, point_b, tangent_vector_a
        )

    def pair_mean(self, point_a, point_b):
        return self._dispatch("pair_mean")(point_a, point_b)

    def zerovec(self, point):
        return self._dispatch("zerovec", reduction=_ProductTangentVector)(
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
