import pytest

from pymanopt.manifolds import PoincareBall
from pymanopt.numerics import NumericsBackend


class TestPoincareBallManifold:
    @pytest.fixture(autouse=True)
    def setup(
        self, real_numerics_backend: NumericsBackend, product_dimension: int
    ):
        self.n = 50
        self.k = product_dimension
        self.backend = real_numerics_backend
        self.manifold = PoincareBall(self.n, k=self.k, backend=self.backend)

    def test_dim(self):
        assert self.manifold.dim == self.k * self.n

    def test_conformal_factor(self):
        x = self.manifold.random_point() / 2
        self.backend.assert_allclose(
            1 - 2 / self.manifold.conformal_factor(x),
            self.backend.linalg_norm(x, axis=-1, keepdims=True) ** 2,
        )

    def test_inner_product(self):
        x = self.manifold.random_point() / 2
        u = self.manifold.random_tangent_vector(x)
        v = self.manifold.random_tangent_vector(x)
        self.backend.assert_allclose(
            self.backend.sum(
                (2 / (1 - self.backend.linalg_norm(x, axis=-1) ** 2)) ** 2
                * self.backend.sum(u * v, axis=-1)
            ),
            self.manifold.inner_product(x, u, v),
        )

        # Test that angles are preserved.
        x = self.manifold.random_point() / 2
        u = self.manifold.random_tangent_vector(x)
        v = self.manifold.random_tangent_vector(x)
        cos_eangle = (
            self.backend.sum(u * v)
            / self.backend.linalg_norm(u)
            / self.backend.linalg_norm(v)
        )
        cos_rangle = (
            self.manifold.inner_product(x, u, v)
            / self.manifold.norm(x, u)
            / self.manifold.norm(x, v)
        )
        self.backend.assert_allclose(cos_rangle, cos_eangle, atol=2e-3)

        # Test symmetry.
        self.backend.assert_allclose(
            self.manifold.inner_product(x, u, v),
            self.manifold.inner_product(x, v, u),
        )

    def test_proj(self):
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)
        self.backend.assert_allclose(u, self.manifold.projection(x, u))

    def test_norm(self):
        # Divide by 2 to avoid round-off errors.
        x = self.manifold.random_point() / 2
        u = self.manifold.random_tangent_vector(x)

        self.backend.assert_allclose(
            self.backend.sum(
                (2 / (1 - self.backend.linalg_norm(x, axis=-1) ** 2)) ** 2
                * self.backend.sum(u * u, axis=-1)
            ),
            self.manifold.norm(x, u) ** 2,
        )

    def test_random_point(self):
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        x = self.manifold.random_point()
        assert self.backend.all(self.backend.linalg_norm(x, axis=-1) < 1)
        y = self.manifold.random_point()
        assert self.backend.all(x != y)

    def test_random_tangent_vector(self):
        # Just make sure that things generated are in the tangent space and
        # that if you generate two they are not equal.
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)
        v = self.manifold.random_tangent_vector(x)

        assert self.backend.all(u != v)

    def test_zero_vector(self):
        x = self.manifold.random_point()
        u = self.manifold.zero_vector(x)
        self.backend.assert_allclose(self.backend.linalg_norm(u), 0)

    def test_dist(self):
        x = self.manifold.random_point() / 2
        y = self.manifold.random_point() / 2
        correct_dist = self.backend.sum(
            self.backend.arccosh(
                1
                + 2
                * self.backend.linalg_norm(x - y, axis=-1) ** 2
                / (1 - self.backend.linalg_norm(x, axis=-1) ** 2)
                / (1 - self.backend.linalg_norm(y, axis=-1) ** 2)
            )
            ** 2
        )
        self.backend.assert_allclose(
            correct_dist, self.manifold.dist(x, y) ** 2
        )

    def test_euclidean_to_riemannian_gradient(self):
        # For now just test whether the method returns an array of the correct
        # shape.
        point = self.manifold.random_point()
        euclidean_gradient = self.backend.random_normal(size=point.shape)
        riemannian_gradient = self.manifold.euclidean_to_riemannian_gradient(
            point, euclidean_gradient
        )
        assert euclidean_gradient.shape == riemannian_gradient.shape

    def test_euclidean_to_riemannian_hessian(self):
        # For now just test whether the method returns an array of the correct
        # shape.
        point = self.manifold.random_point()
        euclidean_gradient = self.backend.random_normal(size=point.shape)
        euclidean_hessian = self.backend.random_normal(size=point.shape)
        tangent_vector = self.manifold.random_tangent_vector(point)
        riemannian_hessian = self.manifold.euclidean_to_riemannian_hessian(
            point, euclidean_gradient, euclidean_hessian, tangent_vector
        )
        assert euclidean_hessian.shape == riemannian_hessian.shape

    def test_retraction(self):
        x = self.manifold.random_point() / 2
        u = self.manifold.random_tangent_vector(x)
        y = self.manifold.retraction(x, u)
        assert self.backend.all(
            self.backend.linalg_norm(y, axis=-1) < 1 + 1e-10
        )

    def test_mobius_addition(self):
        # test if Mobius addition is closed in the Poincare ball
        x = self.manifold.random_point()
        y = self.manifold.random_point()
        z = self.manifold.mobius_addition(x, y)
        # The norm of z may be slightly more than one because of
        # round-off errors.
        assert self.backend.all(
            self.backend.linalg_norm(z, axis=-1) < 1 + 1e-10
        )

    def test_exp_log_inverse(self):
        x = self.manifold.random_point() / 2
        y = self.manifold.random_point() / 2
        explog = self.manifold.exp(x, self.manifold.log(x, y))
        self.backend.assert_allclose(y, explog)

    def test_log_exp_inverse(self):
        x = self.manifold.random_point() / 2
        # If u is too big its exponential will have norm 1 because of
        # numerical approximations
        u = self.manifold.random_tangent_vector(x) / self.manifold.dim
        logexp = self.manifold.log(x, self.manifold.exp(x, u))
        self.backend.assert_allclose(u, logexp)

    def test_pair_mean(self):
        x = self.manifold.random_point() / 2
        y = self.manifold.random_point() / 2
        z = self.manifold.pair_mean(x, y)
        self.backend.assert_allclose(
            self.manifold.dist(x, z), self.manifold.dist(y, z)
        )
