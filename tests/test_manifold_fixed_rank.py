import unittest

import numpy as np
import numpy.linalg as la
import numpy.random as rnd
import numpy.testing as np_testing

from pymanopt.manifolds import FixedRankEmbedded


class TestFixedRankEmbeddedManifold(unittest.TestCase):
    def setUp(self):
        self.m = m = 10
        self.n = n = 5
        self.k = k = 3
        self.man = FixedRankEmbedded(m, n, k)

    def test_dim(self):
        assert self.man.dim == (self.m + self.n - self.k) * self.k

    def test_typicaldist(self):
        assert self.man.dim == self.man.test_typicaldist

    def test_dist(self):
        e = self.man
        x = e.randvec()
        y = e.randvec()
        with self.assertRaises(NotImplementedError):
            e.dist(x, y)

    def test_inner(self):
        e = self.man
        x = e.rand()
        a = e.randvec(x)
        b = e.randvec(x)
        A = x[0].dot(x[1].dot(x[2])) + a[0].dot(x[2].T) + x[0].dot(a[2])
        B = x[0].dot(x[1].dot(x[2])) + b[0].dot(x[2].T) + x[0].dot(b[2])
        trueinner = np.sum(A*B)
        np_testing.assert_almost_equal(trueinner, e.inner(x, a, b))

    def test_proj(self):
        e = self.man
        x = e.rand()
        u = e.randvec(x)
        np_testing.assert_allclose(e.proj(x, u), u)

    def test_norm(self):
        e = self.man
        x = e.rand()
        u = rnd.randn(self.m, self.n)
        np_testing.assert_almost_equal(e.inner(x, u, u), e.norm(x, u))

    def test_rand(self):
        e = self.man
        x = e.rand()
        y = e.rand()
        assert np.shape(x[0]) == (self.m, self.k)
        assert np.shape(x[1]) == (self.k, self.k)
        assert np.shape(x[0]) == (self.n, self.k)
        assert la.norm(x - y) > 1e-6

    def test_transp(self):
        e = self.man
        x = e.rand()
        u = e.randvec(x)
        np_testing.assert_allclose(e.proj(x, u), e.transp(x, u))

'''
    def test_ehess2rhess(self):
        e = self.man
        x = e.rand()
        u = e.randvec(x)
        egrad, ehess = rnd.randn(2, self.m, self.n)
        np_testing.assert_allclose(e.ehess2rhess(x, egrad, ehess, u),
                                   ehess)

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

    def test_randvec(self):
        e = self.man
        x = e.rand()
        u = e.randvec(x)
        v = e.randvec(x)
        assert np.shape(u) == (self.m, self.n)
        np_testing.assert_almost_equal(la.norm(u), 1)
        assert la.norm(u - v) > 1e-6

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
'''
