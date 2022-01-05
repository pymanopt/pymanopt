import abc
import functools
import warnings

import numpy as np


class RetrAsExpMixin:
    """Mixin which defers calls to the exponential map to the retraction."""

    def exp(self, Y, U):
        class_name = self.__class__.__name__
        warnings.warn(
            f"Exponential map for manifold '{class_name}' not available. "
            "Using retraction instead.",
            RuntimeWarning,
        )
        return self.retr(Y, U)


class Manifold(metaclass=abc.ABCMeta):
    """Riemannian manifold base class.

    Abstract base class setting out a template for manifold classes.

    Not all methods are required by all solvers.
    In particular, first order gradient based solvers such as
    :mod:`pymanopt.solvers.steepest_descent` and
    :mod:`pymanopt.solvers.conjugate_gradient` require :meth:`egrad2rgrad` to
    be implemented but not :meth:`ehess2rhess`.
    Second order solvers such as :mod:`pymanopt.solvers.trust_regions` will
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
        represent points as tuples or lists of arrays. In this case,
        `point_layout` describes how many elements such tuples/lists contain.
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
    def typicaldist(self):
        """Returns the `scale` of the manifold.

        This is used by the trust-regions solver to determine default initial
        and maximal trust-region radii.
        """
        raise NotImplementedError(
            f"Manifold '{self.__class__.__name__}' does not provide a "
            "'typicaldist' property"
        )

    # Abstract methods that subclasses must implement.

    @abc.abstractmethod
    def inner(self, X, G, H):
        """Inner product between tangent vectors at a point on the manifold.

        The inner product corresponds to the Riemannian metric between two
        tangent vectors ``G`` and ``H`` in the tangent space at ``X``.
        """

    @abc.abstractmethod
    def proj(self, X, G):
        """Projects vector in the ambient space on the tangent space."""

    @abc.abstractmethod
    def norm(self, X, G):
        """Computes the norm of a tangent vector at a point on the manifold."""

    @abc.abstractmethod
    def rand(self):
        """Returns a random point on the manifold."""

    @abc.abstractmethod
    def randvec(self, X):
        """Returns a random vector in the tangent space at ``X``."""

    @abc.abstractmethod
    def zerovec(self, X):
        """Returns the zero vector in the tangent space at ``X``."""

    # Methods which are only required by certain solvers.

    def _raise_not_implemented_error(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            raise NotImplementedError(
                f"Manifold '{self.__class__.__name__}' provides no "
                f"implementation for '{method.__name__}'"
            )

        return wrapper

    @_raise_not_implemented_error
    def dist(self, X, Y):
        """The geodesic distance between two points on the manifold."""

    @_raise_not_implemented_error
    def egrad2rgrad(self, X, G):
        """Converts the Euclidean to the Riemannian gradient.

        For embedded submanifolds, this is simply the projection of ``G`` on
        the tangent space at ``X``.
        """

    @_raise_not_implemented_error
    def ehess2rhess(self, X, G, H, U):
        """Converts the Euclidean to the Riemannian Hessian.

        This converts the Euclidean Hessian ``H`` of a function at a point
        ``X`` along a tangent vector ``U`` to the Riemannian Hessian of ``X``
        along ``U`` on the manifold.
        """

    @_raise_not_implemented_error
    def retr(self, X, G):
        """Retracts a tangent vector back to the manifold.

        This generalizes the exponential map, and is often more efficient to
        compute numerically.
        It maps a vector ``G`` in the tangent space at ``X`` back to the
        manifold.
        """

    @_raise_not_implemented_error
    def exp(self, X, U):
        """Computes the exponential map on the manifold."""

    @_raise_not_implemented_error
    def log(self, X, Y):
        """Computes the logarithmic map on the manifold.

        This is the inverse of :meth:`exp`.
        """

    @_raise_not_implemented_error
    def transp(self, X1, X2, G):
        """Transport a tangent vector between different tangent spaces.

        The vector transport generalizes the concept of parallel transport, and
        is often more efficient to compute numerically.
        It transports a vector ``G`` in the tangent space at ``X1`` to the
        tangent space at `X2`.
        """

    @_raise_not_implemented_error
    def pairmean(self, X, Y):
        """Computes the intrinsic mean of two points on the manifold.

        Returns the intrinsic mean of two points ``X`` and ``Y`` on the
        manifold, i.e., a point that lies mid-way between ``X`` and ``Y`` on
        the geodesic arc joining them.
        """


class EuclideanEmbeddedSubmanifold(Manifold, metaclass=abc.ABCMeta):
    """Embedded submanifolds of Euclidean space.

    This class provides a generic way to project Euclidean gradients to their
    Riemannian counterparts via the :meth:`egrad2rgrad` method.
    Similarly, if the Weingarten map (also known as shape operator) is provided
    via implementing the :meth:`weingarten` method, the class provides a
    generic implementation of the :meth:`ehess2rhess` method required by
    second-order solvers to translate Euclidean Hessian-vector products to
    their Riemannian counterparts.

    Notes:
        Refer to [AMT2013]_ for the exact definition of the Weingarten map.
    """

    def egrad2rgrad(self, X, G):
        return self.proj(X, G)

    def ehess2rhess(self, X, G, H, U):
        normal_gradient = G - self.proj(X, G)
        return self.proj(X, H) + self.weingarten(X, U, normal_gradient)

    @Manifold._raise_not_implemented_error
    def weingarten(self, X, U, V):
        """Compute the Weingarten map of the manifold.

        This map takes a vector ``U`` in the tangent space at ``X`` and a
        vector ``V`` in the normal space at ``X`` to produce another tangent
        vector.
        """
