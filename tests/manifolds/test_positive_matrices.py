import autograd.numpy as np
import pytest
from numpy import testing as np_testing

from pymanopt.manifolds import Positive


class TestPositiveVectors:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.m = m = 3
        self.n = n = 1
        self.k = k = 2
        self.manifold = Positive(m, n, k=k)

    def test_inner_product(self):
        x = self.manifold.random_point()
        g = self.manifold.random_tangent_vector(x)
        h = self.manifold.random_tangent_vector(x)
        assert self.manifold.inner_product(
            x, g, h
        ) == self.manifold.inner_product(x, h, g)

    def test_norm(self):
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)
        assert self.manifold.norm(x, u) > 0

    def test_random_point(self):
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
        assert (np.linalg.norm(g - h, axis=(1, 2)) > 1e-6).all()

    def test_dist(self):
        # To implement norm of log(x, y)
        x = self.manifold.random_point()
        y = self.manifold.random_point()
        u = self.manifold.log(x, y)
        np_testing.assert_almost_equal(
            self.manifold.norm(x, u), self.manifold.dist(x, y)
        )

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
