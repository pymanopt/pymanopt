import autograd.numpy as np
from numpy import linalg as la, random as rnd, testing as np_testing

from pymanopt.manifolds import Grassmann
from pymanopt.tools import testing
from pymanopt.tools.multi import multieye, multiprod, multisym, multitransp
from .._test import TestCase


class TestSingleGrassmannManifold(TestCase):
    def setUp(self):
        self.m = m = 5
        self.n = n = 2
        self.k = k = 1
        self.man = Grassmann(m, n, k=k)

        self.proj = lambda x, u: u - np.dot(x, np.dot(x.T, u))

    def test_dist(self):
        x = self.man.rand()
        y = self.man.rand()
        np_testing.assert_almost_equal(self.man.dist(x, y),
                                       self.man.norm(x, self.man.log(x, y)))

    def test_ehess2rhess(self):
        # Test this function at some randomly generated point.
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

        np_testing.assert_allclose(multiprod(multitransp(xretru), xretru),
                                   np.eye(self.n),
                                   atol=1e-10)

        u = u * 1e-6
        xretru = self.man.retr(x, u)
        np_testing.assert_allclose(xretru, x + u)

    # def test_egrad2rgrad(self):

    # def test_norm(self):

    def test_rand(self):
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        X = self.man.rand()
        np_testing.assert_allclose(multiprod(multitransp(X), X),
                                   np.eye(self.n), atol=1e-10)
        Y = self.man.rand()
        assert la.norm(X - Y) > 1e-6

    # def test_randvec(self):

    # def test_transp(self):

    def test_exp_log_inverse(self):
        s = self.man
        x = s.rand()
        y = s.rand()
        u = s.log(x, y)
        z = s.exp(x, u)
        np_testing.assert_almost_equal(0, self.man.dist(y, z), decimal=5)

    def test_log_exp_inverse(self):
        s = self.man
        x = s.rand()
        u = s.randvec(x)
        y = s.exp(x, u)
        v = s.log(x, y)
        # Check that the manifold difference between the tangent vectors u and
        # v is 0
        np_testing.assert_almost_equal(0, self.man.norm(x, u - v))

    # def test_pairmean(self):
        # s = self.man
        # X = s.rand()
        # Y = s.rand()
        # Z = s.pairmean(X, Y)
        # np_testing.assert_array_almost_equal(s.dist(X, Z), s.dist(Y, Z))


class TestMultiGrassmannManifold(TestCase):
    def setUp(self):
        self.m = m = 5
        self.n = n = 2
        self.k = k = 3
        self.man = Grassmann(m, n, k=k)

        self.proj = lambda x, u: u - np.dot(x, np.dot(x.T, u))

    def test_dim(self):
        assert self.man.dim == self.k * (self.m * self.n - self.n ** 2)

    def test_typicaldist(self):
        np_testing.assert_almost_equal(self.man.typicaldist,
                                       np.sqrt(self.n * self.k))

    def test_dist(self):
        x = self.man.rand()
        y = self.man.rand()
        np_testing.assert_almost_equal(self.man.dist(x, y),
                                       self.man.norm(x, self.man.log(x, y)))

    def test_inner(self):
        X = self.man.rand()
        A = self.man.randvec(X)
        B = self.man.randvec(X)
        np_testing.assert_allclose(np.sum(A * B), self.man.inner(X, A, B))

    def test_proj(self):
        # Construct a random point X on the manifold.
        X = self.man.rand()

        # Construct a vector H in the ambient space.
        H = rnd.randn(self.k, self.m, self.n)

        # Compare the projections.
        Hproj = H - multiprod(X, multiprod(multitransp(X), H))
        np_testing.assert_allclose(Hproj, self.man.proj(X, H))

    def test_retr(self):
        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        x = self.man.rand()
        u = self.man.randvec(x)

        xretru = self.man.retr(x, u)

        np_testing.assert_allclose(multiprod(multitransp(xretru), xretru),
                                   multieye(self.k, self.n),
                                   atol=1e-10)

        u = u * 1e-6
        xretru = self.man.retr(x, u)
        np_testing.assert_allclose(xretru, x + u)

    # def test_egrad2rgrad(self):

    def test_norm(self):
        x = self.man.rand()
        u = self.man.randvec(x)
        np_testing.assert_almost_equal(self.man.norm(x, u), la.norm(u))

    def test_rand(self):
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        X = self.man.rand()
        np_testing.assert_allclose(multiprod(multitransp(X), X),
                                   multieye(self.k, self.n), atol=1e-10)
        Y = self.man.rand()
        assert la.norm(X - Y) > 1e-6

    def test_randvec(self):
        # Make sure things generated are in tangent space and if you generate
        # two then they are not equal.
        X = self.man.rand()
        U = self.man.randvec(X)
        np_testing.assert_allclose(multisym(multiprod(multitransp(X), U)),
                                   np.zeros((self.k, self.n, self.n)),
                                   atol=1e-10)
        V = self.man.randvec(X)
        assert la.norm(U - V) > 1e-6

    # def test_transp(self):

    def test_exp_log_inverse(self):
        s = self.man
        x = s.rand()
        y = s.rand()
        u = s.log(x, y)
        z = s.exp(x, u)
        np_testing.assert_almost_equal(0, self.man.dist(y, z))

    def test_log_exp_inverse(self):
        s = self.man
        x = s.rand()
        u = s.randvec(x)
        y = s.exp(x, u)
        v = s.log(x, y)
        # Check that the manifold difference between the tangent vectors u and
        # v is 0
        np_testing.assert_almost_equal(0, self.man.norm(x, u - v))

    # def test_pairmean(self):
        # s = self.man
        # X = s.rand()
        # Y = s.rand()
        # Z = s.pairmean(X, Y)
        # np_testing.assert_array_almost_equal(s.dist(X, Z), s.dist(Y, Z))
