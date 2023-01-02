import autograd.numpy as np
import pytest
from numpy import testing as np_testing

from pymanopt.manifolds import PoincareBall


class TestSinglePoincareBallManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.n = 50
        self.manifold = PoincareBall(self.n)

    def test_dim(self):
        assert self.manifold.dim == self.n

    def test_conformal_factor(self):
        x = self.manifold.random_point() / 2
        np_testing.assert_allclose(
            1 - 2 / self.manifold.conformal_factor(x), np.linalg.norm(x) ** 2
        )

    def test_inner_product(self):
        x = self.manifold.random_point() / 2
        u = self.manifold.random_tangent_vector(x)
        v = self.manifold.random_tangent_vector(x)
        np_testing.assert_allclose(
            (2 / (1 - np.linalg.norm(x) ** 2)) ** 2 * np.inner(u, v),
            self.manifold.inner_product(x, u, v),
        )

        # Test that angles are preserved.
        x = self.manifold.random_point() / 2
        u = self.manifold.random_tangent_vector(x)
        v = self.manifold.random_tangent_vector(x)
        cos_eangle = np.sum(u * v) / np.linalg.norm(u) / np.linalg.norm(v)
        cos_rangle = (
            self.manifold.inner_product(x, u, v)
            / self.manifold.norm(x, u)
            / self.manifold.norm(x, v)
        )
        np_testing.assert_allclose(cos_rangle, cos_eangle)

        # Test symmetry.
        np_testing.assert_allclose(
            self.manifold.inner_product(x, u, v),
            self.manifold.inner_product(x, v, u),
        )

    def test_proj(self):
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)
        np_testing.assert_allclose(u, self.manifold.projection(x, u))

    def test_norm(self):
        x = self.manifold.random_point() / 2
        u = self.manifold.random_tangent_vector(x)

        np_testing.assert_allclose(
            2 / (1 - np.linalg.norm(x) ** 2) * np.linalg.norm(u),
            self.manifold.norm(x, u),
        )

    def test_random_point(self):
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        x = self.manifold.random_point()
        np_testing.assert_array_less(np.linalg.norm(x), 1)
        y = self.manifold.random_point()
        assert not np.array_equal(x, y)

    def test_random_tangent_vector(self):
        # Just make sure that things generated are in the tangent space and
        # that if you generate two they are not equal.
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)
        v = self.manifold.random_tangent_vector(x)

        assert not np.array_equal(u, v)

    def test_zero_vector(self):
        x = self.manifold.random_point()
        u = self.manifold.zero_vector(x)
        np_testing.assert_allclose(np.linalg.norm(u), 0)

    def test_dist(self):
        x = self.manifold.random_point() / 2
        y = self.manifold.random_point() / 2
        correct_dist = np.arccosh(
            1
            + 2
            * np.linalg.norm(x - y) ** 2
            / (1 - np.linalg.norm(x) ** 2)
            / (1 - np.linalg.norm(y) ** 2)
        )
        np_testing.assert_allclose(correct_dist, self.manifold.dist(x, y))

    def test_euclidean_to_riemannian_gradient(self):
        # For now just test whether the method returns an array of the correct
        # shape.
        point = self.manifold.random_point()
        euclidean_gradient = np.random.normal(size=point.shape)
        riemannian_gradient = self.manifold.euclidean_to_riemannian_gradient(
            point, euclidean_gradient
        )
        assert euclidean_gradient.shape == riemannian_gradient.shape

    def test_euclidean_to_riemannian_hessian(self):
        # For now just test whether the method returns an array of the correct
        # shape.
        point = self.manifold.random_point()
        euclidean_gradient = np.random.normal(size=point.shape)
        euclidean_hessian = np.random.normal(size=point.shape)
        tangent_vector = self.manifold.random_tangent_vector(point)
        riemannian_hessian = self.manifold.euclidean_to_riemannian_hessian(
            point, euclidean_gradient, euclidean_hessian, tangent_vector
        )
        assert euclidean_hessian.shape == riemannian_hessian.shape

    def test_retraction(self):
        x = self.manifold.random_point() / 2
        u = self.manifold.random_tangent_vector(x)
        y = self.manifold.retraction(x, u)
        assert np.linalg.norm(y) <= 1 + 1e-10

    def test_mobius_addition(self):
        # test if Mobius addition is closed in the Poincare ball
        x = self.manifold.random_point() / 2
        y = self.manifold.random_point() / 2
        z = self.manifold.mobius_addition(x, y)
        # The norm of z may be slightly more than one because of
        # round-off errors.
        assert np.linalg.norm(z) <= 1 + 1e-10

    def test_exp_log_inverse(self):
        x = self.manifold.random_point() / 2
        y = self.manifold.random_point() / 2
        explog = self.manifold.exp(x, self.manifold.log(x, y))
        np_testing.assert_allclose(y, explog)

    def test_log_exp_inverse(self):
        x = self.manifold.random_point() / 2
        # If u is too big its exponential will have norm 1 because of
        # numerical approximations
        u = self.manifold.random_tangent_vector(x) / self.manifold.dim
        logexp = self.manifold.log(x, self.manifold.exp(x, u))
        np_testing.assert_allclose(u, logexp)

    def test_pair_mean(self):
        x = self.manifold.random_point() / 2
        y = self.manifold.random_point() / 2
        z = self.manifold.pair_mean(x, y)
        np_testing.assert_allclose(
            self.manifold.dist(x, z), self.manifold.dist(y, z)
        )


class TestMultiplePoincareBallManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.n = 50
        self.k = 20
        self.manifold = PoincareBall(self.n, k=self.k)

    def test_dim(self):
        assert self.manifold.dim == self.k * self.n

    def test_conformal_factor(self):
        x = self.manifold.random_point() / 2
        np_testing.assert_allclose(
            1 - 2 / self.manifold.conformal_factor(x),
            np.linalg.norm(x, axis=-1, keepdims=True) ** 2,
        )

    def test_inner_product(self):
        x = self.manifold.random_point() / 2
        u = self.manifold.random_tangent_vector(x)
        v = self.manifold.random_tangent_vector(x)
        np_testing.assert_allclose(
            np.sum(
                (2 / (1 - np.linalg.norm(x, axis=-1) ** 2)) ** 2
                * np.sum(u * v, axis=-1)
            ),
            self.manifold.inner_product(x, u, v),
        )

        # Test symmetry.
        np_testing.assert_allclose(
            self.manifold.inner_product(x, u, v),
            self.manifold.inner_product(x, v, u),
        )

    def test_proj(self):
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)
        np_testing.assert_allclose(u, self.manifold.projection(x, u))

    def test_norm(self):
        # Divide by 2 to avoid round-off errors.
        x = self.manifold.random_point() / 2
        u = self.manifold.random_tangent_vector(x)

        np_testing.assert_allclose(
            np.sum(
                (2 / (1 - np.linalg.norm(x, axis=-1) ** 2)) ** 2
                * np.sum(u * u, axis=-1)
            ),
            self.manifold.norm(x, u) ** 2,
        )

    def test_random_point(self):
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        x = self.manifold.random_point()
        np_testing.assert_array_less(np.linalg.norm(x, axis=-1), 1)
        y = self.manifold.random_point()
        assert not np.array_equal(x, y)

    def test_random_tangent_vector(self):
        # Just make sure that things generated are in the tangent space and
        # that if you generate two they are not equal.
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)
        v = self.manifold.random_tangent_vector(x)

        assert not np.array_equal(u, v)

    def test_zero_vector(self):
        x = self.manifold.random_point()
        u = self.manifold.zero_vector(x)
        np_testing.assert_allclose(np.linalg.norm(u), 0)

    def test_dist(self):
        x = self.manifold.random_point() / 2
        y = self.manifold.random_point() / 2
        correct_dist = np.sum(
            np.arccosh(
                1
                + 2
                * np.linalg.norm(x - y, axis=-1) ** 2
                / (1 - np.linalg.norm(x, axis=-1) ** 2)
                / (1 - np.linalg.norm(y, axis=-1) ** 2)
            )
            ** 2
        )
        np_testing.assert_allclose(correct_dist, self.manifold.dist(x, y) ** 2)

    def test_euclidean_to_riemannian_gradient(self):
        # For now just test whether the method returns an array of the correct
        # shape.
        point = self.manifold.random_point()
        euclidean_gradient = np.random.normal(size=point.shape)
        riemannian_gradient = self.manifold.euclidean_to_riemannian_gradient(
            point, euclidean_gradient
        )
        assert euclidean_gradient.shape == riemannian_gradient.shape

    def test_euclidean_to_riemannian_hessian(self):
        # For now just test whether the method returns an array of the correct
        # shape.
        point = self.manifold.random_point()
        euclidean_gradient = np.random.normal(size=point.shape)
        euclidean_hessian = np.random.normal(size=point.shape)
        tangent_vector = self.manifold.random_tangent_vector(point)
        riemannian_hessian = self.manifold.euclidean_to_riemannian_hessian(
            point, euclidean_gradient, euclidean_hessian, tangent_vector
        )
        assert euclidean_hessian.shape == riemannian_hessian.shape

    def test_retraction(self):
        x = self.manifold.random_point() / 2
        u = self.manifold.random_tangent_vector(x)
        y = self.manifold.retraction(x, u)
        np_testing.assert_array_less(np.linalg.norm(y, axis=-1), 1 + 1e-10)

    def test_mobius_addition(self):
        # test if Mobius addition is closed in the Poincare ball
        x = self.manifold.random_point()
        y = self.manifold.random_point()
        z = self.manifold.mobius_addition(x, y)
        # The norm of z may be slightly more than one because of
        # round-off errors.
        np_testing.assert_array_less(np.linalg.norm(z, axis=-1), 1 + 1e-10)

    def test_exp_log_inverse(self):
        x = self.manifold.random_point() / 2
        y = self.manifold.random_point() / 2
        explog = self.manifold.exp(x, self.manifold.log(x, y))
        np_testing.assert_allclose(y, explog)

    def test_log_exp_inverse(self):
        x = self.manifold.random_point() / 2
        # If u is too big its exponential will have norm 1 because of
        # numerical approximations
        u = self.manifold.random_tangent_vector(x) / self.manifold.dim
        logexp = self.manifold.log(x, self.manifold.exp(x, u))
        np_testing.assert_allclose(u, logexp)

    def test_pair_mean(self):
        x = self.manifold.random_point() / 2
        y = self.manifold.random_point() / 2
        z = self.manifold.pair_mean(x, y)
        np_testing.assert_allclose(
            self.manifold.dist(x, z), self.manifold.dist(y, z)
        )
