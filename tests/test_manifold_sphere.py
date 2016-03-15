import unittest

import numpy as np
import numpy.linalg as la
import numpy.random as rnd
import numpy.testing as np_testing

import autograd.numpy as npa

from pymanopt.manifolds import Sphere
import pymanopt.tools.testing as testing


class TestSphereManifold(unittest.TestCase):
    def setUp(self):
        self.m = m = 100
        self.n = n = 50
        self.man = Sphere(m, n)

        # For automatic testing of ehess2rhess
        self.proj = lambda x, u: u - npa.tensordot(x, u, np.ndim(u)) * x

    def test_name(self):
        man = self.man
        m = self.m
        n = self.n
        assert "Sphere manifold of " + str(m) + "x" + str(n) in str(man)

    def test_dim(self):
        assert self.man.dim == self.m * self.n - 1

    def test_typicaldist(self):
        np_testing.assert_almost_equal(self.man.typicaldist, np.pi)

    def test_dist(self):
        s = self.man
        x = s.rand()
        y = s.rand()
        correct_dist = np.arccos(np.tensordot(x, y))
        np.testing.assert_almost_equal(correct_dist, s.dist(x, y))

    def test_inner(self):
        s = self.man
        x = s.rand()
        u = s.randvec(x)
        v = s.randvec(x)
        np.testing.assert_almost_equal(np.sum(u * v), s.inner(x, u, v))

    def test_proj(self):
        #  Construct a random point X on the manifold.
        X = rnd.randn(self.m, self.n)
        X /= la.norm(X, "fro")

        #  Construct a vector H in the ambient space.
        H = rnd.randn(self.m, self.n)

        #  Compare the projections.
        np_testing.assert_array_almost_equal(H - X * np.trace(X.T.dot(H)),
                                             self.man.proj(X, H))

    def test_egrad2rgrad(self):
        # Should be the same as proj
        #  Construct a random point X on the manifold.
        X = rnd.randn(self.m, self.n)
        X /= la.norm(X, "fro")

        #  Construct a vector H in the ambient space.
        H = rnd.randn(self.m, self.n)

        #  Compare the projections.
        np_testing.assert_array_almost_equal(H - X * np.trace(X.T.dot(H)),
                                             self.man.egrad2rgrad(X, H))

    def test_ehess2rhess(self):
        x = self.man.rand()
        u = self.man.randvec(x)
        egrad = rnd.randn(self.m, self.n)
        ehess = rnd.randn(self.m, self.n)

        np_testing.assert_allclose(testing.ehess2rhess(self.proj)(x, egrad,
                                                                  ehess, u),
                                   self.man.ehess2rhess(x, egrad, ehess, u))

    def test_retr(self):
        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        x = self.man.rand()
        u = self.man.randvec(x)

        xretru = self.man.retr(x, u)
        np_testing.assert_almost_equal(la.norm(xretru), 1)

        u = u * 1e-6
        xretru = self.man.retr(x, u)
        np_testing.assert_allclose(xretru, x + u)

    def test_norm(self):
        x = self.man.rand()
        u = self.man.randvec(x)

        np_testing.assert_almost_equal(self.man.norm(x, u), la.norm(u))

    def test_rand(self):
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        s = self.man
        x = s.rand()
        np_testing.assert_almost_equal(la.norm(x), 1)
        y = s.rand()
        assert np.linalg.norm(x - y) > 1e-3

    def test_randvec(self):
        # Just make sure that things generated are in the tangent space and
        # that if you generate two they are not equal.
        s = self.man
        x = s.rand()
        u = s.randvec(x)
        v = s.randvec(x)
        np_testing.assert_almost_equal(np.tensordot(x, u), 0)

        assert np.linalg.norm(u - v) > 1e-3

    def test_transp(self):
        # Should be the same as proj
        s = self.man
        x = s.rand()
        y = s.rand()
        u = s.randvec(x)

        np_testing.assert_allclose(s.transp(x, y, u), s.proj(y, u))

    def test_exp_log_inverse(self):
        s = self.man
        X = s.rand()
        U = s.randvec(X)
        Uexplog = s.exp(X, s.log(X, U))
        np_testing.assert_array_almost_equal(U, Uexplog)

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


class TestSphereManifoldVector(unittest.TestCase):
    def setUp(self):
        self.n = n = 50
        self.man = Sphere(n)

    def test_name(self):
        man = self.man
        n = self.n
        assert "Sphere manifold of " + str(n) + "-vectors" in str(man)


class TestSphereManifoldTensor(unittest.TestCase):
    def setUp(self):
        self.n1 = n1 = 100
        self.n2 = n2 = 50
        self.n3 = n3 = 25
        self.man = Sphere(n1, n2, n3)

    def test_name(self):
        man = self.man
        n1 = self.n1
        n2 = self.n2
        n3 = self.n3
        assert ("Sphere manifold of shape (" + str(n1) + ", " + str(n2) +
                ", " + str(n3)) in str(man)
