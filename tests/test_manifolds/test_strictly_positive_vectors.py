import numpy as np
from numpy import linalg as la
from numpy import random as rnd
from numpy import testing as np_testing

from pymanopt.manifolds import StrictlyPositiveVectors

# from pymanopt.tools import testing
from .._test import TestCase


class TestSingleStrictlyPositiveVectors(TestCase):
    def setUp(self):
        self.n = n = 10
        self.k = k = 1
        self.man = StrictlyPositiveVectors(n, k=k)

    def test_inner(self):
        x = self.man.rand()
        g = 2 * self.man.randvec(x)
        h = 3 * self.man.randvec(x)
        assert type(self.man.inner(x, g, h)) is np.float64

    def test_proj(self):
        # Test proj(proj(X)) == proj(X)
        x = self.man.rand()
        u = rnd.randn(self.n)
        proj_u = self.man.proj(x, u)
        proj_proj_u = self.man.proj(x, proj_u)

        np_testing.assert_allclose(proj_u, proj_proj_u)

    def test_norm(self):
        x = self.man.rand()
        u = self.man.randvec(x)
        x_u = (1.0 / x) * u
        assert type(self.man.norm(x, u)) is np.float64
        np_testing.assert_allclose(la.norm(x_u), self.man.norm(x, u))

    def test_rand(self):
        # Just make sure that things generated are on the manifold
        # and that if you generate two they are not equal.
        n = self.n
        x = self.man.rand()
        assert x.shape == (n, 1)
        assert (x > 0).all()
        y = self.man.rand()
        assert (self.man.dist(x, y)).all() > 1e-6

    def test_randvec(self):
        # Just make sure that if you generate two they are not equal.
        # check also if unit norm
        n = self.n
        x = self.man.rand()
        g = self.man.randvec(x)
        assert g.shape == (n, 1)
        h = self.man.randvec(x)
        assert (la.norm(g - h, axis=0) > 1e-6).all()
        np_testing.assert_allclose(self.man.norm(x, g), 1)

    def test_zerovec(self):
        x = self.man.rand()
        np_testing.assert_allclose(
            self.man.zerovec(x),
            np.zeros((self.n, 1))
        )

    def test_dist(self):
        # To implement norm of log(x, y)
        x = self.man.rand()
        y = self.man.rand()
        u = self.man.log(x, y)
        assert type(self.man.dist(x, y)) is np.float64
        np_testing.assert_allclose(self.man.norm(x, u),
                                   self.man.dist(x, y))

    def test_ehess2rhess(self):
        n = self.n
        x = self.man.rand()
        u = self.man.randvec(x)
        egrad = rnd.normal(size=(n, 1))
        ehess = rnd.normal(size=(n, 1))
        hess = self.man.ehess2rhess(x, egrad, ehess, u)
        hess_proj = self.man.proj(x, hess)

        np_testing.assert_allclose(hess, hess_proj)

    def test_exp_log_inverse(self):
        x = self.man.rand()
        y = self.man.rand()
        u = self.man.log(x, y)
        z = self.man.exp(x, u)
        np_testing.assert_allclose(self.man.dist(y, z), 0, atol=1e-10)

    def test_log_exp_inverse(self):
        x = self.man.rand()
        u = self.man.randvec(x)
        y = self.man.exp(x, u)
        v = self.man.log(x, y)
        np_testing.assert_allclose(self.man.norm(x, u - v), 0, atol=1e-10)

    def test_retr(self):
        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        x = self.man.rand()
        u = self.man.randvec(x)

        xretru = self.man.retr(x, u)

        assert (xretru > 0).all()

        u = u * 1e-6
        xretru = self.man.retr(x, u)
        np_testing.assert_allclose(xretru, x + u)

    def test_transp(self):
        # check that vector remains in tangent space
        m = self.man
        x = m.rand()
        y = m.rand()
        u = m.randvec(x)
        t_u = m.transp(x, y, u)
        np_testing.assert_allclose(t_u, m.proj(y, t_u))


class TestProductStrictlyPositiveVectors():
    def __init__(self):
        self.n = n = 10
        self.k = k = 2
        self.man = StrictlyPositiveVectors(n, k=k)

    def test_inner(self):
        x = self.man.rand()
        g = 2 * self.man.randvec(x)
        h = 3 * self.man.randvec(x)
        assert type(self.man.inner(x, g, h)) is np.float64

    def test_proj(self):
        # Test proj(proj(X)) == proj(X)
        x = self.man.rand()
        u = rnd.randn(self.n)
        proj_u = self.man.proj(x, u)
        proj_proj_u = self.man.proj(x, proj_u)

        np_testing.assert_allclose(proj_u, proj_proj_u)

    def test_norm(self):
        x = self.man.rand()
        u = self.man.randvec(x)
        x_u = (1.0 / x) * u
        assert type(self.man.norm(x, u)) is np.float64
        np_testing.assert_allclose(la.norm(x_u), self.man.norm(x, u))

    def test_rand(self):
        # Just make sure that things generated are on the manifold
        # and that if you generate two they are not equal.
        k, n = self.k, self.n
        x = self.man.rand()
        assert x.shape == (k, n, 1)
        assert (x > 0).all()
        y = self.man.rand()
        assert (self.man.dist(x, y)).all() > 1e-6

    def test_randvec(self):
        # Just make sure that if you generate two they are not equal.
        # check also if unit norm
        k, n = self.k, self.n
        x = self.man.rand()
        g = self.man.randvec(x)
        assert g.shape == (k, n, 1)
        h = self.man.randvec(x)
        assert (la.norm(g - h, axis=0) > 1e-6).all()
        np_testing.assert_allclose(self.man.norm(x, g), 1)

    def test_zerovec(self):
        x = self.man.rand()
        np_testing.assert_allclose(
            self.man.zerovec(x),
            np.zeros((self.k, self.n, 1))
        )

    def test_dist(self):
        # To implement norm of log(x, y)
        x = self.man.rand()
        y = self.man.rand()
        u = self.man.log(x, y)
        assert type(self.man.dist(x, y)) is np.float64
        np_testing.assert_allclose(self.man.norm(x, u),
                                   self.man.dist(x, y))

    def test_ehess2rhess(self):
        k, n = self.k, self.n
        x = self.man.rand()
        u = self.man.randvec(x)
        egrad = rnd.normal(size=(k, n, 1))
        ehess = rnd.normal(size=(k, n, 1))
        hess = self.man.ehess2rhess(x, egrad, ehess, u)
        hess_proj = self.man.proj(x, hess)

        np_testing.assert_allclose(hess, hess_proj)

    def test_exp_log_inverse(self):
        x = self.man.rand()
        y = self.man.rand()
        u = self.man.log(x, y)
        z = self.man.exp(x, u)
        np_testing.assert_allclose(self.man.dist(y, z), 0, atol=1e-10)

    def test_log_exp_inverse(self):
        x = self.man.rand()
        u = self.man.randvec(x)
        y = self.man.exp(x, u)
        v = self.man.log(x, y)
        np_testing.assert_allclose(self.man.norm(x, u - v), 0, atol=1e-10)

    def test_retr(self):
        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        x = self.man.rand()
        u = self.man.randvec(x)

        xretru = self.man.retr(x, u)

        assert (xretru > 0).all()

        u = u * 1e-6
        xretru = self.man.retr(x, u)
        np_testing.assert_allclose(xretru, x + u)

    def test_transp(self):
        # check that vector remains in tangent space
        m = self.man
        x = m.rand()
        y = m.rand()
        u = m.randvec(x)
        t_u = m.transp(x, y, u)
        np_testing.assert_allclose(t_u, m.proj(y, t_u))
