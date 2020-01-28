import numpy as np
from numpy import linalg as la, random as rnd, testing as np_testing

from pymanopt.manifolds import Symmetric
from pymanopt.tools.multi import multisym
from .._test import TestCase


class TestSymmetricManifold(TestCase):
    def setUp(self):
        self.n = n = 10
        self.k = k = 5
        self.man = Symmetric(n, k)

    def test_dim(self):
        assert self.man.dim == 0.5 * self.k * self.n * (self.n + 1)

    def test_typicaldist(self):
        man = self.man
        np_testing.assert_almost_equal(man.typicaldist, np.sqrt(man.dim))

    def test_dist(self):
        e = self.man
        x, y = rnd.randn(2, self.k, self.n, self.n)
        np_testing.assert_almost_equal(e.dist(x, y), la.norm(x - y))

    def test_inner(self):
        e = self.man
        x = e.rand()
        y = e.randvec(x)
        z = e.randvec(x)
        np_testing.assert_almost_equal(np.sum(y * z), e.inner(x, y, z))

    def test_proj(self):
        e = self.man
        x = e.rand()
        u = np.random.randn(self.k, self.n, self.n)
        np_testing.assert_allclose(e.proj(x, u), multisym(u))

    def test_ehess2rhess(self):
        e = self.man
        x = e.rand()
        u = e.randvec(x)
        egrad, ehess = rnd.randn(2, self.k, self.n, self.n)
        np_testing.assert_allclose(e.ehess2rhess(x, egrad, ehess, u),
                                   multisym(ehess))

    def test_retr(self):
        e = self.man
        x = e.rand()
        u = e.randvec(x)
        np_testing.assert_allclose(e.retr(x, u), x + u)

    def test_egrad2rgrad(self):
        e = self.man
        x = e.rand()
        u = e.randvec(x)
        np_testing.assert_allclose(e.egrad2rgrad(x, u), u)

    def test_norm(self):
        e = self.man
        x = e.rand()
        u = rnd.randn(self.n, self.n, self.k)
        np_testing.assert_almost_equal(np.sqrt(np.sum(u**2)), e.norm(x, u))

    def test_rand(self):
        e = self.man
        x = e.rand()
        y = e.rand()
        assert np.shape(x) == (self.k, self.n, self.n)
        np_testing.assert_allclose(x, multisym(x))
        assert la.norm(x - y) > 1e-6

    def test_randvec(self):
        e = self.man
        x = e.rand()
        u = e.randvec(x)
        v = e.randvec(x)
        assert np.shape(u) == (self.k, self.n, self.n)
        np_testing.assert_allclose(u, multisym(u))
        np_testing.assert_almost_equal(la.norm(u), 1)
        assert la.norm(u - v) > 1e-6

    def test_transp(self):
        e = self.man
        x = e.rand()
        y = e.rand()
        u = e.randvec(x)
        np_testing.assert_allclose(e.transp(x, y, u), u)

    def test_exp_log_inverse(self):
        s = self.man
        X = s.rand()
        Y = s.rand()
        Yexplog = s.exp(X, s.log(X, Y))
        np_testing.assert_array_almost_equal(Y, Yexplog)

    def test_log_exp_inverse(self):
        s = self.man
        X = s.rand()
        U = s.randvec(X)
        Ulogexp = s.log(X, s.exp(X, U))
        np_testing.assert_array_almost_equal(U, Ulogexp)

    def test_pairmean(self):
        s = self.man
        X = s.rand()
        Y = s.rand()
        Z = s.pairmean(X, Y)
        np_testing.assert_array_almost_equal(s.dist(X, Z), s.dist(Y, Z))
