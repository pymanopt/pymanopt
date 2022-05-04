import autograd.numpy as np
from numpy import testing as np_testing

from pymanopt.manifolds import StrictlyPositiveVectors

# from pymanopt.tools import testing
from .._test import TestCase


class TestStrictlyPositiveVectors(TestCase):
    def setUp(self):
        self.n = n = 3
        self.k = k = 2
        self.manifold = StrictlyPositiveVectors(n, k=k)

    def test_inner_product(self):
        x = self.manifold.random_point()
        g = self.manifold.random_tangent_vector(x)
        h = self.manifold.random_tangent_vector(x)
        assert (
            self.manifold.inner_product(x, g, h).shape == np.array([1, self.k])
        ).all()

    def test_projection(self):
        # Test proj(proj(X)) == proj(X)
        x = self.manifold.random_point()
        u = np.random.normal(size=self.n)
        proj_u = self.manifold.projection(x, u)
        proj_proj_u = self.manifold.projection(x, proj_u)

        np_testing.assert_allclose(proj_u, proj_proj_u)

    def test_norm(self):
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)
        x_u = (1.0 / x) * u
        np_testing.assert_almost_equal(
            np.linalg.norm(x_u, axis=0, keepdims=True),
            self.manifold.norm(x, u),
        )

    def test_rand(self):
        # Just make sure that things generated are on the manifold
        # and that if you generate two they are not equal.
        x = self.manifold.random_point()
        assert (x > 0).all()
        y = self.manifold.random_point()
        assert (self.manifold.dist(x, y)).all() > 1e-6

    def test_random_tangent_vector(self):
        # Just make sure that if you generate two they are not equal.
        # check also if unit norm
        x = self.manifold.random_point()
        g = self.manifold.random_tangent_vector(x)
        h = self.manifold.random_tangent_vector(x)
        assert (np.linalg.norm(g - h, axis=0) > 1e-6).all()
        np_testing.assert_almost_equal(self.manifold.norm(x, g), 1)

    def test_dist(self):
        # To implement norm of log(x, y)
        x = self.manifold.random_point()
        y = self.manifold.random_point()
        u = self.manifold.log(x, y)
        np_testing.assert_almost_equal(
            self.manifold.norm(x, u), self.manifold.dist(x, y)
        )

    # def test_euclidean_to_riemannian_hessian(self):

    def test_exp_log_inverse(self):
        x = self.manifold.random_point()
        y = self.manifold.random_point()
        u = self.manifold.log(x, y)
        z = self.manifold.exp(x, u)
        np_testing.assert_almost_equal(self.manifold.dist(y, z), 0)

    def test_log_exp_inverse(self):
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)
        y = self.manifold.exp(x, u)
        v = self.manifold.log(x, y)
        np_testing.assert_almost_equal(self.manifold.norm(x, u - v), 0)

    def test_retraction(self):
        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)

        xretru = self.manifold.retraction(x, u)

        assert (xretru > 0).all()

        u = u * 1e-6
        xretru = self.manifold.retraction(x, u)
        np_testing.assert_allclose(xretru, x + u)

        # def test_transport(self):
