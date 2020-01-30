import autograd.numpy as np
from numpy import linalg as la, random as rnd, testing as np_testing

from pymanopt.manifolds import Stiefel
from pymanopt.tools import testing
from pymanopt.tools.multi import multieye, multiprod, multisym, multitransp
from .._test import TestCase


class TestSingleStiefelManifold(TestCase):
    def setUp(self):
        self.m = m = 20
        self.n = n = 2
        self.k = k = 1
        self.man = Stiefel(m, n, k=k)
        self.proj = lambda x, u: u - np.dot(x, np.dot(x.T, u) +
                                            np.dot(u.T, x)) / 2

    def test_dim(self):
        assert self.man.dim == 0.5 * self.n * (2 * self.m - self.n - 1)

    # def test_typicaldist(self):

    # def test_dist(self):

    def test_inner(self):
        X = la.qr(rnd.randn(self.m, self.n))[0]
        A, B = rnd.randn(2, self.m, self.n)
        np_testing.assert_allclose(np.sum(A * B), self.man.inner(X, A, B))

    def test_proj(self):
        # Construct a random point X on the manifold.
        X = rnd.randn(self.m, self.n)
        X = la.qr(X)[0]

        # Construct a vector H in the ambient space.
        H = rnd.randn(self.m, self.n)

        # Compare the projections.
        Hproj = H - X.dot(X.T.dot(H) + H.T.dot(X)) / 2
        np_testing.assert_allclose(Hproj, self.man.proj(X, H))

    def test_rand(self):
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        X = self.man.rand()
        np_testing.assert_allclose(X.T.dot(X), np.eye(self.n), atol=1e-10)
        Y = self.man.rand()
        assert np.linalg.norm(X - Y) > 1e-6

    def test_randvec(self):
        # Make sure things generated are in tangent space and if you generate
        # two then they are not equal.
        X = self.man.rand()
        U = self.man.randvec(X)
        np_testing.assert_allclose(multisym(X.T.dot(U)),
                                   np.zeros((self.n, self.n)), atol=1e-10)
        V = self.man.randvec(X)
        assert la.norm(U - V) > 1e-6

    def test_retr(self):
        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        x = self.man.rand()
        u = self.man.randvec(x)

        xretru = self.man.retr(x, u)
        np_testing.assert_allclose(xretru.T.dot(xretru), np.eye(self.n,
                                                                self.n),
                                   atol=1e-10)

        u = u * 1e-6
        xretru = self.man.retr(x, u)
        np_testing.assert_allclose(xretru, x + u)

    def test_ehess2rhess(self):
        # Test this function at some randomly generated point.
        x = self.man.rand()
        u = self.man.randvec(x)
        egrad = rnd.randn(self.m, self.n)
        ehess = rnd.randn(self.m, self.n)

        np_testing.assert_allclose(testing.ehess2rhess(self.proj)(x, egrad,
                                                                  ehess, u),
                                   self.man.ehess2rhess(x, egrad, ehess, u))

    # def test_egrad2rgrad(self):

    def test_norm(self):
        x = self.man.rand()
        u = self.man.randvec(x)
        np_testing.assert_almost_equal(self.man.norm(x, u), la.norm(u))

    # def test_transp(self):

    def test_exp(self):
        # Check that exp lies on the manifold and that exp of a small vector u
        # is close to x + u.
        s = self.man
        x = s.rand()
        u = s.randvec(x)

        xexpu = s.exp(x, u)
        np_testing.assert_allclose(xexpu.T.dot(xexpu), np.eye(self.n,
                                                              self.n),
                                   atol=1e-10)

        u = u * 1e-6
        xexpu = s.exp(x, u)
        np_testing.assert_allclose(xexpu, x + u)

    # def test_exp_log_inverse(self):
        # s = self.man
        # X = s.rand()
        # U = s.randvec(X)
        # Uexplog = s.exp(X, s.log(X, U))
        # np_testing.assert_array_almost_equal(U, Uexplog)

    # def test_log_exp_inverse(self):
        # s = self.man
        # X = s.rand()
        # U = s.randvec(X)
        # Ulogexp = s.log(X, s.exp(X, U))
        # np_testing.assert_array_almost_equal(U, Ulogexp)

    # def test_pairmean(self):
        # s = self.man
        # X = s.rand()
        # Y = s.rand()
        # Z = s.pairmean(X, Y)
        # np_testing.assert_array_almost_equal(s.dist(X, Z), s.dist(Y, Z))


class TestMultiStiefelManifold(TestCase):
    def setUp(self):
        self.m = m = 10
        self.n = n = 3
        self.k = k = 3
        self.man = Stiefel(m, n, k=k)

    def test_dim(self):
        assert self.man.dim == 0.5 * self.k * self.n * (2 * self.m - self.n -
                                                        1)

    def test_typicaldist(self):
        np_testing.assert_almost_equal(self.man.typicaldist,
                                       np.sqrt(self.n * self.k))

    # def test_dist(self):

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
        Hproj = H - multiprod(X, multiprod(multitransp(X), H) +
                              multiprod(multitransp(H), X)) / 2
        np_testing.assert_allclose(Hproj, self.man.proj(X, H))

    def test_rand(self):
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        X = self.man.rand()
        np_testing.assert_allclose(multiprod(multitransp(X), X),
                                   multieye(self.k, self.n), atol=1e-10)
        Y = self.man.rand()
        assert np.linalg.norm(X - Y) > 1e-6

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

    # def test_transp(self):

    def test_exp(self):
        # Check that exp lies on the manifold and that exp of a small vector u
        # is close to x + u.
        s = self.man
        x = s.rand()
        u = s.randvec(x)

        xexpu = s.exp(x, u)
        np_testing.assert_allclose(multiprod(multitransp(xexpu), xexpu),
                                   multieye(self.k, self.n), atol=1e-10)

        u = u * 1e-6
        xexpu = s.exp(x, u)
        np_testing.assert_allclose(xexpu, x + u)

    # def test_exp_log_inverse(self):
        # s = self.man
        # X = s.rand()
        # U = s.randvec(X)
        # Uexplog = s.exp(X, s.log(X, U))
        # np_testing.assert_array_almost_equal(U, Uexplog)

    # def test_log_exp_inverse(self):
        # s = self.man
        # X = s.rand()
        # U = s.randvec(X)
        # Ulogexp = s.log(X, s.exp(X, U))
        # np_testing.assert_array_almost_equal(U, Ulogexp)

    # def test_pairmean(self):
        # s = self.man
        # X = s.rand()
        # Y = s.rand()
        # Z = s.pairmean(X, Y)
        # np_testing.assert_array_almost_equal(s.dist(X, Z), s.dist(Y, Z))
