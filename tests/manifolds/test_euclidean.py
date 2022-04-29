import numpy as np
from numpy import linalg as la
from numpy import testing as np_testing

from pymanopt.manifolds import Euclidean

from .._test import TestCase


class TestEuclideanManifold(TestCase):
    def setUp(self):
        self.m = m = 10
        self.n = n = 5
        self.man = Euclidean(m, n)

    def test_dim(self):
        assert self.man.dim == self.m * self.n

    def test_typical_dist(self):
        np_testing.assert_almost_equal(
            self.man.typical_dist, np.sqrt(self.m * self.n)
        )

    def test_dist(self):
        e = self.man
        x, y = np.random.randn(2, self.m, self.n)
        np_testing.assert_almost_equal(e.dist(x, y), la.norm(x - y))

    def test_inner(self):
        e = self.man
        x = e.rand()
        y = e.random_tangent_vector(x)
        z = e.random_tangent_vector(x)
        np_testing.assert_almost_equal(np.sum(y * z), e.inner(x, y, z))

    def test_projection(self):
        e = self.man
        x = e.rand()
        u = e.random_tangent_vector(x)
        np_testing.assert_allclose(e.projection(x, u), u)

    def test_ehess2rhess(self):
        e = self.man
        x = e.rand()
        u = e.random_tangent_vector(x)
        egrad, ehess = np.random.randn(2, self.m, self.n)
        np_testing.assert_allclose(e.ehess2rhess(x, egrad, ehess, u), ehess)

    def test_retraction(self):
        e = self.man
        x = e.rand()
        u = e.random_tangent_vector(x)
        np_testing.assert_allclose(e.retraction(x, u), x + u)

    def test_egrad2rgrad(self):
        e = self.man
        x = e.rand()
        u = e.random_tangent_vector(x)
        np_testing.assert_allclose(e.egrad2rgrad(x, u), u)

    def test_norm(self):
        e = self.man
        x = e.rand()
        u = np.random.randn(self.m, self.n)
        np_testing.assert_almost_equal(np.sqrt(np.sum(u**2)), e.norm(x, u))

    def test_rand(self):
        e = self.man
        x = e.rand()
        y = e.rand()
        assert np.shape(x) == (self.m, self.n)
        assert la.norm(x - y) > 1e-6

    def test_random_tangent_vector(self):
        e = self.man
        x = e.rand()
        u = e.random_tangent_vector(x)
        v = e.random_tangent_vector(x)
        assert np.shape(u) == (self.m, self.n)
        np_testing.assert_almost_equal(la.norm(u), 1)
        assert la.norm(u - v) > 1e-6

    def test_transport(self):
        e = self.man
        x = e.rand()
        y = e.rand()
        u = e.random_tangent_vector(x)
        np_testing.assert_allclose(e.transport(x, y, u), u)

    def test_exp_log_inverse(self):
        s = self.man
        X = s.rand()
        Y = s.rand()
        Yexplog = s.exp(X, s.log(X, Y))
        np_testing.assert_array_almost_equal(Y, Yexplog)

    def test_log_exp_inverse(self):
        s = self.man
        X = s.rand()
        U = s.random_tangent_vector(X)
        Ulogexp = s.log(X, s.exp(X, U))
        np_testing.assert_array_almost_equal(U, Ulogexp)

    def test_pair_mean(self):
        s = self.man
        X = s.rand()
        Y = s.rand()
        Z = s.pair_mean(X, Y)
        np_testing.assert_array_almost_equal(s.dist(X, Z), s.dist(Y, Z))
