import autograd.numpy as np
from numpy import linalg as la, random as rnd, testing as np_testing

from pymanopt.manifolds import StrictlyPositiveVectors
# from pymanopt.tools import testing
from .._test import TestCase


class TestStrictlyPositiveVectors(TestCase):
    def setUp(self):
        self.n = n = 3
        self.k = k = 2
        self.man = StrictlyPositiveVectors(n, k=k)

    def test_inner(self):
        x = self.man.rand()
        g = self.man.randvec(x)
        h = self.man.randvec(x)
        assert (self.man.inner(x, g, h).shape == np.array([1, self.k])).all()

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
        x_u = (1./x) * u
        np_testing.assert_almost_equal(
            la.norm(x_u, axis=0, keepdims=True),
            self.man.norm(x, u))

    def test_rand(self):
        # Just make sure that things generated are on the manifold
        # and that if you generate two they are not equal.
        x = self.man.rand()
        assert (x > 0).all()
        y = self.man.rand()
        assert (self.man.dist(x, y)).all() > 1e-6

    def test_randvec(self):
        # Just make sure that if you generate two they are not equal.
        # check also if unit norm
        x = self.man.rand()
        g = self.man.randvec(x)
        h = self.man.randvec(x)
        assert (la.norm(g-h, axis=0) > 1e-6).all()
        np_testing.assert_almost_equal(self.man.norm(x, g), 1)

    def test_dist(self):
        # To implement norm of log(x, y)
        x = self.man.rand()
        y = self.man.rand()
        u = self.man.log(x, y)
        np_testing.assert_almost_equal(self.man.norm(x, u),
                                       self.man.dist(x, y))

    # def test_ehess2rhess(self):

    def test_exp_log_inverse(self):
        x = self.man.rand()
        y = self.man.rand()
        u = self.man.log(x, y)
        z = self.man.exp(x, u)
        np_testing.assert_almost_equal(self.man.dist(y, z), 0)

    def test_log_exp_inverse(self):
        x = self.man.rand()
        u = self.man.randvec(x)
        y = self.man.exp(x, u)
        v = self.man.log(x, y)
        np_testing.assert_almost_equal(self.man.norm(x, u - v), 0)

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

        # def test_transp(self):
