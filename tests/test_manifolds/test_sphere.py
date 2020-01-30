import warnings

import autograd.numpy as np
from numpy import linalg as la, random as rnd, testing as np_testing

from pymanopt.manifolds import (Sphere, SphereSubspaceComplementIntersection,
                                SphereSubspaceIntersection)
from pymanopt.tools import testing
from .._test import TestCase


class TestSphereManifold(TestCase):
    def setUp(self):
        self.m = m = 100
        self.n = n = 50
        self.man = Sphere(m, n)

        # For automatic testing of ehess2rhess
        self.proj = lambda x, u: u - np.tensordot(x, u, np.ndim(u)) * x

    def test_dim(self):
        assert self.man.dim == self.m * self.n - 1

    def test_typicaldist(self):
        np_testing.assert_almost_equal(self.man.typicaldist, np.pi)

    def test_dist(self):
        s = self.man
        x = s.rand()
        y = s.rand()
        correct_dist = np.arccos(np.tensordot(x, y))
        np_testing.assert_almost_equal(correct_dist, s.dist(x, y))

    def test_inner(self):
        s = self.man
        x = s.rand()
        u = s.randvec(x)
        v = s.randvec(x)
        np_testing.assert_almost_equal(np.sum(u * v), s.inner(x, u, v))

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


class TestSphereSubspaceIntersectionManifold(TestCase):
    def setUp(self):
        self.n = 2
        # Defines the 1-sphere intersected with the 1-dimensional subspace
        # passing through (1, 1) / sqrt(2). This creates a 0-dimensional
        # manifold as it only consits of isolated points in R^2.
        self.U = np.ones((self.n, 1)) / np.sqrt(2)
        with warnings.catch_warnings(record=True):
            self.man = SphereSubspaceIntersection(self.U)

    def test_dim(self):
        self.assertEqual(self.man.dim, 0)

    def test_rand(self):
        x = self.man.rand()
        p = np.ones(2) / np.sqrt(2)
        # The manifold only consists of two isolated points (cf. `setUp()`).
        self.assertTrue(np.allclose(x, p) or np.allclose(x, -p))

    def test_proj(self):
        h = rnd.randn(self.n)
        x = self.man.rand()
        p = self.man.proj(x, h)
        # Since the manifold is 0-dimensional, the tangent at each point is
        # simply the 0-dimensional space {0}.
        np_testing.assert_array_almost_equal(p, np.zeros(self.n))

    def test_dim_1(self):
        U = np.zeros((3, 2))
        U[0, 0] = U[1, 1] = 1
        man = SphereSubspaceIntersection(U)
        # U spans the x-y plane, therefore the manifold consists of the
        # 1-sphere in the x-y plane, and has dimension 1.
        self.assertEqual(man.dim, 1)
        # Check if a random element from the manifold has vanishing
        # z-component.
        x = man.rand()
        np_testing.assert_almost_equal(x[-1], 0)

    def test_dim_rand(self):
        n = 100
        U = rnd.randn(n, n // 3)
        dim = la.matrix_rank(U) - 1
        man = SphereSubspaceIntersection(U)
        self.assertEqual(man.dim, dim)


class TestSphereSubspaceComplementIntersectionManifold(TestCase):
    def setUp(self):
        self.n = 2
        # Define the 1-sphere intersected with the 1-dimensional subspace
        # orthogonal to the line passing through (1, 1) / sqrt(2). This creates
        # a 0-dimensional manifold as it only consits of isolated points in
        # R^2.
        self.U = np.ones((self.n, 1)) / np.sqrt(2)
        with warnings.catch_warnings(record=True):
            self.man = SphereSubspaceComplementIntersection(self.U)

    def test_dim(self):
        self.assertEqual(self.man.dim, 0)

    def test_rand(self):
        x = self.man.rand()
        p = np.array([-1, 1]) / np.sqrt(2)
        self.assertTrue(np.allclose(x, p) or np.allclose(x, -p))

    def test_proj(self):
        h = rnd.randn(self.n)
        x = self.man.rand()
        p = self.man.proj(x, h)
        # Since the manifold is 0-dimensional, the tangent at each point is
        # simply the 0-dimensional space {0}.
        np_testing.assert_array_almost_equal(p, np.zeros(self.n))

    def test_dim_1(self):
        U = np.zeros((3, 1))
        U[-1, -1] = 1
        man = SphereSubspaceComplementIntersection(U)
        # U spans the z-axis with its orthogonal complement being the x-y
        # plane, therefore the manifold consists of the 1-sphere in the x-y
        # plane, and has dimension 1.
        self.assertEqual(man.dim, 1)
        # Check if a random element from the manifold has vanishing
        # z-component.
        x = man.rand()
        np_testing.assert_almost_equal(x[-1], 0)

    def test_dim_rand(self):
        n = 100
        U = rnd.randn(n, n // 3)
        # By the rank-nullity theorem the orthogonal complement of span(U) has
        # dimension n - rank(U).
        dim = n - la.matrix_rank(U) - 1
        man = SphereSubspaceComplementIntersection(U)
        self.assertEqual(man.dim, dim)

        # Test if a random element really lies in the left null space of U.
        x = man.rand()
        np_testing.assert_almost_equal(la.norm(x), 1)
        np_testing.assert_array_almost_equal(U.T.dot(x), np.zeros(U.shape[1]))
