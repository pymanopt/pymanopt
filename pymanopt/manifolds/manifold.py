import abc
import functools
import warnings

import numpy as np


class RetrAsExpMixin:
    """Mixin which defers calls to the exponential map to the retraction."""

    def exp(self, point, tangent_vector):
        class_name = self.__class__.__name__
        warnings.warn(
            f"Exponential map for manifold '{class_name}' not available. "
            "Using retraction instead.",
            RuntimeWarning,
        )
        return self.retraction(point, tangent_vector)


class Manifold(metaclass=abc.ABCMeta):
    """Riemannian manifold base class.

    Abstract base class setting out a template for manifold classes.

    Not all methods are required by all optimizers.
    In particular, first order gradient based optimizers such as
    :mod:`pymanopt.optimizers.steepest_descent` and
    :mod:`pymanopt.optimizers.conjugate_gradient` require :meth:`egrad2rgrad` to
    be implemented but not :meth:`ehess2rhess`.
    Second order optimizers such as :mod:`pymanopt.optimizers.trust_regions` will
    require :meth:`ehess2rhess`.
    """

    def __init__(self, name, dimension, point_layout=1):
        if not isinstance(dimension, (int, np.integer)):
            raise TypeError("Manifold dimension must be of type int")
        if dimension < 0:
            raise ValueError("Manifold dimension must be positive")
        if not isinstance(point_layout, (int, tuple, list)):
            raise TypeError(
                "Point layout must be of type int, tuple or list, not "
                f"{type(point_layout)}"
            )
        if isinstance(point_layout, (tuple, list)):
            if not all([num_arguments > 0 for num_arguments in point_layout]):
                raise ValueError(
                    f"Invalid point layout {point_layout}: all values must be "
                    "positive"
                )
        elif point_layout <= 0:
            raise ValueError(
                f"Invalid point layout {point_layout}: must be positive"
            )

        self._name = name
        self._dimension = dimension
        self._point_layout = point_layout

    def __str__(self):
        return self._name

    @property
    def dim(self):
        """The dimension of the manifold."""
        return self._dimension

    @property
    def point_layout(self):
        """The number of elements a point on a manifold consists of.

        For most manifolds, which represent points as (potentially
        multi-dimensional) arrays, this will be 1, but other manifolds might
        represent points as tuples or lists of arrays.
        In this case, `point_layout` describes how many elements such
        tuples/lists contain.
        """
        return self._point_layout

    @property
    def num_values(self):
        """Total number of values representing a point on the manifold."""
        if isinstance(self.point_layout, (tuple, list)):
            return sum(self.point_layout)
        return self.point_layout

    # Manifold properties that subclasses can define

    @property
    def typical_dist(self):
        """Returns the `scale` of the manifold.

        This is used by the trust-regions optimizer to determine default
        initial and maximal trust-region radii.
        """
        raise NotImplementedError(
            f"Manifold '{self.__class__.__name__}' does not provide a "
            "'typical_dist' property"
        )

    # Abstract methods that subclasses must implement.

    @abc.abstractmethod
    def inner(
        self,
        point: np.ndarray,
        tangent_vector_a: np.ndarray,
        tangent_vector_b: np.ndarray,
    ) -> np.float64:
        """Inner product between tangent vectors at a point on the manifold.

        This method implements a Riemannian inner product between two tangent
        vectors ``tangent_vector_a`` and ``tangent_vector_b`` in the tangent
        space at ``point``.

        Args:
            point: The base point.
            tangent_vector_a: The first tangent vector.
            tangent_vector_b: The second tangent vector.

        Returns:
            The inner product between ``tangent_vector_a`` and
            ``tangent_vector_b`` in the tangent space at ``point``.
        """

    @abc.abstractmethod
    def projection(self, point, vector):
        """Projects vector in the ambient space on the tangent space."""

    @abc.abstractmethod
    def norm(self, point, tangent_vector):
        """Computes the norm of a tangent vector at a point on the manifold."""

    @abc.abstractmethod
    def rand(self):
        """Returns a random point on the manifold."""

    @abc.abstractmethod
    def randvec(self, point):
        """Returns a random vector in the tangent space at ``point``."""

    @abc.abstractmethod
    def zero_vector(self, point):
        """Returns the zero vector in the tangent space at ``point``."""

    # Methods which are only required by certain optimizers.

    def _raise_not_implemented_error(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            raise NotImplementedError(
                f"Manifold '{self.__class__.__name__}' provides no "
                f"implementation for '{method.__name__}'"
            )

        return wrapper

    @_raise_not_implemented_error
    def dist(self, point_a, point_b):
        """The geodesic distance between two points on the manifold."""

    @_raise_not_implemented_error
    def egrad2rgrad(self, point, euclidean_gradient):
        """Converts the Euclidean to the Riemannian gradient.

        For embedded submanifolds of Euclidean space, this is simply the
        projection of ``euclidean_gradient`` on the tangent space at ``point``.
        """

    @_raise_not_implemented_error
    def ehess2rhess(
        self, point, euclidean_gradient, euclidean_hvp, tangent_vector
    ):
        """Converts the Euclidean to the Riemannian Hessian.

        This converts the Euclidean Hessian-vector product (hvp)
        ``euclidean_hvp`` of a function at a point ``point`` along a tangent
        vector ``tangent_vector`` to the Riemannian hvp of ``point`` along
        ``tangent_vector`` on the manifold.
        """

    @_raise_not_implemented_error
    def retraction(self, point, tangent_vector):
        """Retracts a tangent vector back to the manifold.

        This generalizes the exponential map, and is often more efficient to
        compute numerically.
        It maps a vector ``tangent_vector`` in the tangent space at ``point``
        back to the manifold.
        """

    @_raise_not_implemented_error
    def exp(self, point, tangent_vector):
        """Computes the exponential map on the manifold."""

    @_raise_not_implemented_error
    def log(self, point_a, point_b):
        """Computes the logarithmic map on the manifold.

        The logarithmic map ``log(point_a, point_b)`` produces a tangent vector
        in the tangent space at ``point_a`` that points in the direction of
        ``point_b``.
        In other words, ``exp(point_a, log(point_a, point_b)) == point_b``.
        As such it is the inverse of :meth:`exp`.
        """

    @_raise_not_implemented_error
    def transport(self, point_a, point_b, tangent_vector_a):
        """Compute transport of tangent vectors between tangent spaces.

        This may either be a vector transport (a generalization of parallel
        transport) as defined in section 8.1 of [AMS2008]_, or a transporter
        (see e.g. section 10.5 of [Bou2020]_).
        It transports a vector ``tangent_vector_a`` in the tangent space at
        ``point_a`` to the tangent space at `point_b`.
        """

    @_raise_not_implemented_error
    def pair_mean(self, point_a, point_b):
        """Computes the intrinsic mean of two points on the manifold.

        Returns the intrinsic mean of two points ``X`` and ``Y`` on the
        manifold, i.e., a point that lies mid-way between ``X`` and ``Y`` on
        the geodesic arc joining them.
        """

    @_raise_not_implemented_error
    def to_tangent_space(self, point, vector):
        """Re-tangentialize a vector.

        This method guarantees that ``vector`` is indeed a tangent vector
        at ``point`` on the manifold.
        Typically this simply corresponds to ``proj(point, vector)`` but may
        differ for certain manifolds.
        """


class EuclideanEmbeddedSubmanifold(Manifold, metaclass=abc.ABCMeta):
    """Embedded submanifolds of Euclidean space.

    This class provides a generic way to project Euclidean gradients to their
    Riemannian counterparts via the :meth:`egrad2rgrad` method.
    Similarly, if the Weingarten map (also known as shape operator) is provided
    via implementing the :meth:`weingarten` method, the class provides a
    generic implementation of the :meth:`ehess2rhess` method required by
    second-order optimizers to translate Euclidean Hessian-vector products to
    their Riemannian counterparts.

    Notes:
        Refer to [AMT2013]_ for the exact definition of the Weingarten map.
    """

    def egrad2rgrad(self, point, euclidean_gradient):
        return self.projection(point, euclidean_gradient)

    def ehess2rhess(
        self, point, euclidean_gradient, euclidean_hvp, tangent_vector
    ):
        normal_gradient = euclidean_gradient - self.projection(
            point, euclidean_gradient
        )
        return self.projection(point, euclidean_hvp) + self.weingarten(
            point, tangent_vector, normal_gradient
        )

    @Manifold._raise_not_implemented_error
    def weingarten(self, point, tangent_vector, normal_vector):
        """Compute the Weingarten map of the manifold.

        This map takes a vector ``tangent_vector`` in the tangent space at
        ``point`` and a vector ``normal_vector`` in the normal space at
        ``point`` to produce another tangent vector.
        """
