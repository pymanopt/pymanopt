from pymanopt.manifolds.manifold import Manifold
from pymanopt.numerics import DummyNumericsBackendSingleton, NumericsBackend


class PoincareBall(Manifold):
    r"""The Poincare ball.

    The Poincare ball of dimension ``n``.
    Elements are represented as arrays of shape ``(n,)`` if ``k = 1``.
    For ``k > 1``, the class represents the product manifold of ``k`` Poincare
    balls of dimension ``n``, in which case points are represented as arrays of
    shape ``(k, n)``.

    Since the manifold is open, the tangent space at every point is a copy of
    :math:`\R^n`.

    The Poincare ball is embedded in :math:`\R^n` and is a Riemannian manifold,
    but it is not an embedded Riemannian submanifold since the metric is not
    inherited from the Euclidean inner product of its ambient space.
    Instead, the Riemannian metric is conformal to the Euclidean one (angles are
    preserved), and it is given at every point :math:`\vmx` by
    :math:`\inner{\vmu}{\vmv}_\vmx = \lambda_\vmx^2 \inner{\vmu}{\vmv}` where
    :math:`\lambda_\vmx = 2 / (1 - \norm{\vmx}^2)` is the conformal factor.
    This induces the following distance between two points :math:`\vmx` and
    :math:`\vmy` on the manifold:

        :math:`\dist_\manM(\vmx, \vmy) = \arccosh\parens{1 + 2 \frac{\norm{\vmx
        - \vmy}^2}{(1 - \norm{\vmx}^2) (1 - \norm{\vmy}^2)}}.`

    The norm here is understood as the Euclidean norm in the ambient space.

    Args:
        n: The dimension of the Poincare ball.
        k: The number of elements in the product of Poincare balls.
    """

    def __init__(
        self,
        n: int,
        *,
        k: int = 1,
        backend: NumericsBackend = DummyNumericsBackendSingleton,
    ):
        self._n = n
        self._k = k

        if n < 1:
            raise ValueError(f"Need n >= 1. Value given was n = {n}")
        if k < 1:
            raise ValueError(f"Need k >= 1. Value given was k = {k}")

        if k == 1:
            name = f"Poincare ball B({n})"
        elif k >= 2:
            name = f"Poincare ball B({n})^{k}"

        dimension = k * n
        super().__init__(name, dimension, backend=backend)

    @property
    def typical_dist(self):
        return self.dim / 8

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        factor = self.conformal_factor(point)
        return self.backend.tensordot(
            tangent_vector_a,
            tangent_vector_b * factor**2,
            axes=self.backend.ndim(tangent_vector_a),
        )

    def projection(self, point, vector):
        return vector

    to_tangent_space = projection

    def norm(self, point, tangent_vector):
        return self.backend.sqrt(
            self.inner_product(point, tangent_vector, tangent_vector)
        )

    def random_point(self):
        array = self.backend.random_normal(size=(self._k, self._n))
        norm = self.backend.linalg_norm(array, axis=-1, keepdims=True)
        radius = self.backend.random_uniform(size=(self._k, 1)) ** (
            1.0 / self._n
        )
        point = array / norm * radius
        if self._k == 1:
            return point[0]
        return point

    def random_tangent_vector(self, point):
        vector = self.backend.random_normal(size=point.shape)
        return vector / self.norm(point, vector)

    def zero_vector(self, point):
        return self.backend.zeros_like(point)

    def dist(self, point_a, point_b):
        norm_point_a = self.backend.linalg_norm(point_a, axis=-1) ** 2
        norm_point_b = self.backend.linalg_norm(point_b, axis=-1) ** 2
        norm_difference = (
            self.backend.linalg_norm(point_a - point_b, axis=-1) ** 2
        )
        return self.backend.linalg_norm(
            self.backend.arccosh(
                1
                + 2
                * norm_difference
                / ((1 - norm_point_a) * (1 - norm_point_b))
            )
        )

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        # The hyperbolic metric tensor is conformal to the Euclidean one, so
        # the Euclidean gradient is simply rescaled.
        factor = self.conformal_factor(point)
        return euclidean_gradient * (1 / factor**2)

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        # This expression is derived from the Koszul formula.
        factor = self.conformal_factor(point)
        return (
            self.backend.sum(
                euclidean_gradient * point, axis=-1, keepdims=True
            )
            * tangent_vector
            - self.backend.sum(point * tangent_vector, axis=-1, keepdims=True)
            * euclidean_gradient
            - self.backend.sum(
                euclidean_gradient * tangent_vector, axis=-1, keepdims=True
            )
            * point
            + euclidean_hessian / factor
        ) / factor

    def exp(self, point, tangent_vector):
        norm_point = self.backend.linalg_norm(
            tangent_vector, axis=-1, keepdims=True
        )
        factor = self.conformal_factor(point)
        return self.mobius_addition(
            point,
            tangent_vector
            * (
                self.backend.tanh(norm_point * factor / 2)
                / (norm_point + (norm_point == 0))
            ),
        )

    retraction = exp

    def log(self, point_a, point_b):
        w = self.mobius_addition(-point_a, point_b)
        norm_w = self.backend.linalg_norm(w, axis=-1, keepdims=True)
        factor = self.conformal_factor(point_a)
        return self.backend.arctanh(norm_w) * w / norm_w / (factor / 2)

    def pair_mean(self, point_a, point_b):
        return self.exp(point_a, self.log(point_a, point_b) / 2)

    def mobius_addition(self, point_a, point_b):
        """Möbius addition.

        Special non-associative and non-commutative operation which is closed
        in the Poincare ball.

        Args:
            point_a: The first point.
            point_b: The second point.

        Returns:
            The Möbius sum of ``point_a`` and ``point_b``.
        """
        scalar_product = self.backend.sum(
            point_a * point_b, axis=-1, keepdims=True
        )
        norm_point_a = self.backend.sum(
            point_a * point_a, axis=-1, keepdims=True
        )
        norm_point_b = self.backend.sum(
            point_b * point_b, axis=-1, keepdims=True
        )

        return (
            point_a * (1 + 2 * scalar_product + norm_point_b)
            + point_b * (1 - norm_point_a)
        ) / (1 + 2 * scalar_product + norm_point_a * norm_point_b)

    def conformal_factor(self, point):
        """The conformal factor for a point.

        Args:
            point: The point for which to compute the conformal factor.

        Returns:
            The conformal factor.
            If ``point`` is a point on the product manifold of ``k`` Poincare
            balls, the return value will be an array of shape ``(k,1)``.
            The singleton dimension is explicitly kept to simplify
            multiplication of ``point`` by the conformal factor on product
            manifolds.
        """
        return 2 / (
            1 - self.backend.sum(point * point, axis=-1, keepdims=True)
        )
