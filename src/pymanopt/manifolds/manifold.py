import abc
import functools
import warnings
from typing import Sequence, Union
from types import SimpleNamespace

import pymanopt.numerics as nx


class ParametersBackend(SimpleNamespace):
    """A class to store parameters of a manifold.

    It handles the different backends from nx.
    """
    def __init__(self):
        super().__init__()

    def __setattr__(self, name, value):
        value = nx.numpy_to_backend(value, 'numpy')
        super().__setattr__(name, value)

    def update_backend(self, backend):
        for name, value in self.__dict__.items():
            setattr(self, name, nx.numpy_to_backend(value, backend))


def raise_not_implemented_error(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        raise NotImplementedError(
            f"Manifold '{self.__class__.__name__}' provides no "
            f"implementation for '{method.__name__}'"
        )

    return wrapper


class BackendManifold(type):
    def __new__(cls, name, bases, attrs):
        for method in ['random_point', 'zero_vector']:
            if method in attrs:
                attrs[method] = cls._to_numerics_from_self(cls, attrs[method])

        if 'random_tangent_vector' in attrs:
            attrs['random_tangent_vector'] = cls._to_numerics_from_point(
                cls, attrs['random_tangent_vector'])

        return super().__new__(cls, name, bases, attrs)

    def _to_numerics_from_self(cls, method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            point = method(self, *args, **kwargs)

            backend = self.backend
            if backend is None:
                self.backend = backend = 'numpy'

            point = nx.numpy_to_backend(point, backend)

            return point

        return wrapper

    def _to_numerics_from_point(cls, method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            tangent_vec = method(self, *args, **kwargs)

            if len(args) > 0:
                point = args[0]
            else:
                point = kwargs['point']

            tangent_vec = nx.array_as(tangent_vec, as_=point)

            return tangent_vec

        return wrapper


class Manifold(metaclass=BackendManifold):
    """Riemannian manifold base class.

    Abstract base class setting out a template for manifold classes.

    Args:
        name: String representation of the manifold.
        dimension: The dimension of the manifold, i.e., the vector space
            dimension of the tangent spaces.
        point_layout: Abstract description of the representation of points on
            the manifold.
            For manifolds representing points as simple numpy arrays,
            ``point_layout`` is ``1``.
            For more complicated manifolds which might represent points as a
            tuple or list of ``n`` arrays, `point_layout` would be ``n``.
            Finally, in the special case of the
            :class:`pymanopt.manifolds.product.Product` manifold
            ``point_layout`` will be a compound sequence of point layouts of
            manifolds involved in the product.

    Note:
        Not all methods are required by all optimizers.
        In particular, first order gradient based optimizers such as
        :mod:`pymanopt.optimizers.steepest_descent.SteepestDescent` and
        :mod:`pymanopt.optimizers.conjugate_gradient.ConjugateGradient` require
        :meth:`euclidean_to_riemannian_gradient` to be implemented but not
        :meth:`euclidean_to_riemannian_hessian`.
        Second-order optimizers such as
        :class:`pymanopt.optimizers.trust_regions.TrustRegions` will require
        :meth:`euclidean_to_riemannian_hessian`.
    """

    def __init__(
        self,
        name: str,
        dimension: int,
        point_layout: Union[int, Sequence[int]] = 1,
    ):
        if not isinstance(dimension, (int, nx.integer)):
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
        self._backend = None
        self.parameters = ParametersBackend()

    def __str__(self):
        return self._name

    @property
    def dim(self) -> int:
        """The dimension of the manifold."""
        return self._dimension

    @property
    def point_layout(self):
        """The number of elements a point on a manifold consists of.

        For most manifolds, which represent points as (potentially
        multi-dimensional) arrays, this will be 1, but other manifolds might
        represent points as tuples or lists of arrays.
        In this case, :attr:`point_layout` describes how many elements such
        tuples/lists contain.
        """
        return self._point_layout

    @property
    def num_values(self) -> int:
        """Total number of values representing a point on the manifold."""
        if isinstance(self.point_layout, (tuple, list)):
            return sum(self.point_layout)
        return self.point_layout

    # Manifold properties that subclasses can define.

    @property
    def typical_dist(self):
        """Returns the `scale` of the manifold.

        This is used by the trust-regions optimizer to determine default
        initial and maximal trust-region radii.

        Raises:
            NotImplementedError: If no :attr:`typical_dist` is defined.
        """
        raise NotImplementedError(
            f"Manifold '{self.__class__.__name__}' does not provide a "
            "'typical_dist' property"
        )

    # Abstract methods that subclasses must implement.

    @abc.abstractmethod
    def inner_product(
        self,
        point: nx.ndarray,
        tangent_vector_a: nx.ndarray,
        tangent_vector_b: nx.ndarray,
    ) -> float:
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
        """Projects vector in the ambient space on the tangent space.

        Args:
            point: A point on the manifold.
            vector: A vector in the ambient space of the tangent space at
                ``point``.

        Returns:
            An element of the tangent space at ``point`` closest to ``vector``
            in the ambient space.
        """

    @abc.abstractmethod
    def norm(self, point, tangent_vector):
        """Computes the norm of a tangent vector at a point on the manifold.

        Args:
            point: A point on the manifold.
            tangent_vector: A tangent vector in the tangent space at ``point``.

        Returns:
            The norm of ``tangent_vector``.
        """

    @abc.abstractmethod
    def random_point(self):
        """Returns a random point on the manifold.

        Returns:
            A randomly chosen point on the manifold.
        """

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend):
        """Set the backend used by the random_point method.

        Args:
            backend: The backend to use.
        """
        backend = str(backend).lower()
        print(f"Manifold '{self.__class__.__name__}': backend is {backend}")
        if backend not in nx.NUMERICS_SUPPORTED_BACKENDS:
            raise ValueError(
                "Invalid backend '{backend}': must be one of "
                f"{nx.NUMERICS_SUPPORTED_BACKENDS}"
            )
        self._backend = backend

        self.parameters.update_backend(backend)

    @abc.abstractmethod
    def random_tangent_vector(self, point):
        """Returns a random vector in the tangent space at ``point``.

        Args:
            point: A point on the manifold.

        Returns:
            A randomly chosen tangent vector in the tangent space at ``point``.
        """

    @abc.abstractmethod
    def zero_vector(self, point):
        """Returns the zero vector in the tangent space at ``point``.

        Args:
            point: A point on the manifold.

        Returns:
            The origin of the tangent space at ``point``.
        """

    # Methods which are only required by certain optimizers.

    @raise_not_implemented_error
    def dist(self, point_a, point_b):
        """The geodesic distance between two points on the manifold.

        Args:
            point_a: The first point on the manifold.
            point_b: The second point on the manifold.

        Returns:
            The distance between ``point_a`` and ``point_b`` on the manifold.
        """

    @raise_not_implemented_error
    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        """Converts the Euclidean to the Riemannian gradient.

        Args:
            point: The point on the manifold at which the Euclidean gradient
                was evaluated.
            euclidean_gradient: The Euclidean gradient as a vector in the
                ambient space of the tangent space at ``point``.

        Returns:
            The Riemannian gradient at ``point``.
            This must be a tangent vector at ``point``.
        """

    @raise_not_implemented_error
    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        """Converts the Euclidean to the Riemannian Hessian.

        This converts the Euclidean Hessian ``euclidean_hessian`` of a function
        at a point ``point`` along a tangent vector ``tangent_vector`` to the
        Riemannian Hessian of ``point`` along ``tangent_vector`` on the
        manifold.

        Args:
            point: The point on the manifold at which the Euclidean gradient
                and Hessian was evaluated.
            euclidean_gradient: The Euclidean gradient at ``point``.
            euclidean_hessian: The Euclidean Hessian at ``point`` along the
                direction ``tangent_vector``.
            tangent_vector: The tangent vector in the direction of which the
                Riemannian Hessian is to be calculated.

        Returns:
            The Riemannian Hessian as a tangent vector at ``point``.
        """

    @raise_not_implemented_error
    def retraction(self, point, tangent_vector):
        """Retracts a tangent vector back to the manifold.

        This generalizes the exponential map, and is often more efficient to
        compute numerically.
        It maps a vector ``tangent_vector`` in the tangent space at ``point``
        back to the manifold.

        Args:
            point: A point on the manifold.
            tangent_vector: A tangent vector at ``point``.

        Returns:
            A point on the manifold reached by moving away from ``point`` in
            the direction of ``tangent_vector``.
        """

    @raise_not_implemented_error
    def exp(self, point, tangent_vector):
        """Computes the exponential map on the manifold.

        Args:
            point: A point on the manifold.
            tangent_vector: A tangent vector at ``point``.

        Returns:
            The point on the manifold reached by moving away from ``point``
            along a geodesic in the direction of ``tangent_vector``.
        """

    @raise_not_implemented_error
    def log(self, point_a, point_b):
        """Computes the logarithmic map on the manifold.

        The logarithmic map ``log(point_a, point_b)`` produces a tangent vector
        in the tangent space at ``point_a`` that points in the direction of
        ``point_b``.
        In other words, ``exp(point_a, log(point_a, point_b)) == point_b``.
        As such it is the inverse of :meth:`exp`.

        Args:
            point_a: First point on the manifold.
            point_b: Second point on the manifold.

        Returns:
            A tangent vector in the tangent space at ``point_a``.
        """

    @raise_not_implemented_error
    def transport(self, point_a, point_b, tangent_vector_a):
        """Compute transport of tangent vectors between tangent spaces.

        This may either be a vector transport (a generalization of parallel
        transport) as defined in section 8.1 of [AMS2008]_, or a transporter
        (see e.g. section 10.5 of [Bou2020]_).
        It transports a vector ``tangent_vector_a`` in the tangent space at
        ``point_a`` to the tangent space at ``point_b``.

        Args:
            point_a: The first point on the manifold.
            point_b: The second point on the manifold.
            tangent_vector_a: The tangent vector at ``point_a`` to transport to
                the tangent space at ``point_b``.

        Returns:
            A tangent vector at ``point_b``.
        """

    @raise_not_implemented_error
    def pair_mean(self, point_a, point_b):
        """Computes the intrinsic mean of two points on the manifold.

        Returns the intrinsic mean of two points ``point_a`` and ``point_b`` on
        the manifold, i.e., a point that lies mid-way between ``point_a`` and
        ``point_b`` on the geodesic arc joining them.

        Args:
            point_a: The first point on the manifold.
            point_b: The second point on the manifold.

        Returns:
            The mid-way point between ``point_a`` and ``point_b``.
        """

    @raise_not_implemented_error
    def to_tangent_space(self, point, vector):
        """Re-tangentialize a vector.

        This method guarantees that ``vector`` is indeed a tangent vector
        at ``point`` on the manifold.
        Typically this simply corresponds to a call to meth:`projection` but
        may differ for certain manifolds.

        Args:
            point: A point on the manifold.
            vector: A vector close to the tangent space at ``point``.

        Returns:
            The tangent vector at ``point`` closest to ``vector``.
        """

    def embedding(self, point, tangent_vector):
        """Convert tangent vector to ambient space representation.

        Certain manifolds represent tangent vectors in a format that is more
        convenient for numerical calculations than their representation in the
        ambient space.
        Euclidean Hessian operators generally expect tangent vectors in their
        ambient space representation though.
        This method allows switching between the two possible representations,
        For most manifolds, ``embedding`` is simply the identity map.

        Args:
            point: A point on the manifold.
            tangent_vector: A tangent vector in the internal representation of
                the manifold.

        Returns:
            The same tangent vector in the ambient space representation.

        Note:
            This method is mainly needed internally by the
            :class:`pymanopt.core.problem.Problem` class in order to convert
            tangent vectors to the representation expected by user-given or
            autodiff-generated Euclidean Hessian operators.
        """
        return tangent_vector


class RiemannianSubmanifold(Manifold):
    """Base class for Riemannian submanifolds of Euclidean space.

    This class provides a generic method to project Euclidean gradients to
    their Riemannian counterparts via the
    :meth:`euclidean_to_riemannian_gradient` method.
    Similarly, if the Weingarten map (also known as shape operator) is provided
    via implementing the :meth:`weingarten` method, the class provides a
    generic implementation of the :meth:`euclidean_to_riemannian_hessian`
    method required by second-order optimizers to translate Euclidean
    Hessian-vector products to their Riemannian counterparts.

    Note:
        This class follows definition 3.47 in [Bou2020]_ of "Riemannian
        submanifolds".
        As such, manifolds derived from this class are assumed to be embedded
        submanifolds of Euclidean space with the Riemannian metric inherited
        from the embedding space obtained by restricting it to the tangent
        space at a given point.

        For the exact definition of the Weingarten map refer to [AMT2013]_ and
        the notes in section 5.11 of [Bou2020]_.
    """

    @raise_not_implemented_error
    def weingarten(self, point, tangent_vector, normal_vector):
        """Compute the Weingarten map of the manifold.

        This map takes a vector ``tangent_vector`` in the tangent space at
        ``point`` and a vector ``normal_vector`` in the normal space at
        ``point`` to produce another tangent vector.

        Args:
            point: A point on the manifold.
            tangent_vector: A tangent vector at ``point``.
            normal_vector: A vector orthogonal to the tangent space at
                ``point``.

        Returns:
            A tangent vector.
        """

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return self.projection(point, euclidean_gradient)

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        normal_gradient = euclidean_gradient - self.projection(
            point, euclidean_gradient
        )
        return self.projection(point, euclidean_hessian) + self.weingarten(
            point, tangent_vector, normal_gradient
        )


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

    exp.__doc__ = Manifold.exp.__doc__
