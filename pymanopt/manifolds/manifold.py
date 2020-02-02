import abc
import functools

import numpy as np


class Manifold(metaclass=abc.ABCMeta):
    """
    Abstract base class setting out a template for manifold classes. If you
    would like to extend Pymanopt with a new manifold, then your manifold
    should inherit from this class.

    Not all methods are required by all solvers. In particular, first order
    gradient based solvers such as
    :py:mod:`pymanopt.solvers.steepest_descent` and
    :py:mod:`pymanopt.solvers.conjugate_gradient` require
    :py:func:`egrad2rgrad` to be implemented but not :py:func:`ehess2rhess`.
    Second order solvers such as :py:mod:`pymanopt.solvers.trust_regions`
    will require :py:func:`ehess2rhess`.

    All of these methods correspond closely to methods in
    `Manopt <http://www.manopt.org>`_. See
    http://www.manopt.org/tutorial.html#manifolds for more details on manifolds
    in Manopt, which are effectively identical to those in Pymanopt (all of the
    methods in this class have equivalents in Manopt with the same name).
    """

    def __init__(self, name, dimension, point_layout=1):
        assert isinstance(dimension, (int, np.integer)), \
            "dimension must be an integer"
        assert ((isinstance(point_layout, int) and point_layout > 0) or
                (isinstance(point_layout, (list, tuple)) and
                 all(np.array(point_layout) > 0))), \
            ("'point_layout' must be a positive integer or a sequence of "
             "positive integers")

        self._name = name
        self._dimension = dimension
        self._point_layout = point_layout

    def __str__(self):
        """Returns a string representation of the particular manifold."""
        return self._name

    def _get_class_name(self):
        return self.__class__.__name__

    @property
    def dim(self):
        """The dimension of the manifold"""
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

    # Manifold properties that subclasses can define

    @property
    def typicaldist(self):
        """Returns the "scale" of the manifold. This is used by the
        trust-regions solver to determine default initial and maximal
        trust-region radii.
        """
        raise NotImplementedError(
            "Manifold class '{:s}' does not provide a 'typicaldist'".format(
                self._get_class_name()))

    # Abstract methods that subclasses must implement

    @abc.abstractmethod
    def inner(self, X, G, H):
        """Returns the inner product (i.e., the Riemannian metric) between two
        tangent vectors `G` and `H` in the tangent space at `X`.
        """

    @abc.abstractmethod
    def proj(self, X, G):
        """Projects a vector `G` in the ambient space on the tangent space at
        `X`.
        """

    @abc.abstractmethod
    def norm(self, X, G):
        """Computes the norm of a tangent vector `G` in the tangent space at
        `X`.
        """

    @abc.abstractmethod
    def rand(self):
        """Returns a random point on the manifold."""

    @abc.abstractmethod
    def randvec(self, X):
        """Returns a random vector in the tangent space at `X`. This does not
        follow a specific distribution.
        """

    @abc.abstractmethod
    def zerovec(self, X):
        """Returns the zero vector in the tangent space at X."""

    # Methods which are only required by certain solvers

    def _raise_not_implemented_error(method):
        """Method decorator which raises a NotImplementedError with some meta
        information about the manifold and method if a decorated method is
        called.
        """
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            raise NotImplementedError(
                "Manifold class '{:s}' provides no implementation for "
                "'{:s}'".format(self._get_class_name(), method.__name__))
        return wrapper

    @_raise_not_implemented_error
    def dist(self, X, Y):
        """Returns the geodesic distance between two points `X` and `Y` on the
        manifold."""

    @_raise_not_implemented_error
    def egrad2rgrad(self, X, G):
        """Maps the Euclidean gradient `G` in the ambient space on the tangent
        space of the manifold at `X`. For embedded submanifolds, this is simply
        the projection of `G` on the tangent space at `X`.
        """

    @_raise_not_implemented_error
    def ehess2rhess(self, X, G, H, U):
        """Converts the Euclidean gradient `G` and Hessian `H` of a function at
        a point `X` along a tangent vector `U` to the Riemannian Hessian of `X`
        along `U` on the manifold.
        """

    @_raise_not_implemented_error
    def retr(self, X, G):
        """Computes a retraction mapping a vector `G` in the tangent space at
        `X` to the manifold.
        """

    @_raise_not_implemented_error
    def exp(self, X, U):
        """Computes the Lie-theoretic exponential map of a tangent vector `U`
        at `X`.
        """

    @_raise_not_implemented_error
    def log(self, X, Y):
        """Computes the Lie-theoretic logarithm of `Y`. This is the inverse of
        `exp`.
        """

    @_raise_not_implemented_error
    def transp(self, X1, X2, G):
        """Computes a vector transport which transports a vector `G` in the
        tangent space at `X1` to the tangent space at `X2`.
        """

    @_raise_not_implemented_error
    def pairmean(self, X, Y):
        """Returns the intrinsic mean of two points `X` and `Y` on the
        manifold, i.e., a point that lies mid-way between `X` and `Y` on the
        geodesic arc joining them.
        """


class EuclideanEmbeddedSubmanifold(Manifold, metaclass=abc.ABCMeta):
    """A class to model embedded submanifolds of a Euclidean space. It provides
    a generic way to project Euclidean gradients to their Riemannian
    counterparts via the `egrad2rgrad` method. Similarly, if the Weingarten map
    (also known as shape operator) is provided via implementing the
    'weingarten' method, the class provides a generic implementation of the
    'ehess2rhess' method required by second-order solvers to translate
    Euclidean Hessian-vector products to their Riemannian counterparts.

    Notes
    -----
    Refer to [1]_ for the exact definition of the Weingarten map.

    References
    ----------
    .. [1] Absil, P-A., Robert Mahony, and Jochen Trumpf. "An extrinsic look at
       the Riemannian Hessian." International Conference on Geometric Science
       of Information. Springer, Berlin, Heidelberg, 2013.
    """

    def egrad2rgrad(self, X, G):
        return self.proj(X, G)

    def ehess2rhess(self, X, G, H, U):
        """Converts the Euclidean gradient `G` and Hessian `H` of a function at
        a point `X` along a tangent vector `U` to the Riemannian Hessian of `X`
        along `U` on the manifold. This uses the Weingarten map
        """
        normal_gradient = G - self.proj(X, G)
        return self.proj(X, H) + self.weingarten(X, U, normal_gradient)

    @Manifold._raise_not_implemented_error
    def weingarten(self, X, U, V):
        """Evaluates the Weingarten map of the manifold. This map takes a
        vector `U` in the tangent space at `X` and a vector `V` in the
        normal space at `X` to produce another tangent vector.
        """
