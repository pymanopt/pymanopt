import autograd.numpy as np
from numpy import linalg as la, random as rnd, testing as np_testing

from pymanopt.manifolds import ComplexGrassmann
# from pymanopt.tools import testing
from pymanopt.tools.multi import multieye, multihconj, multiprod, multisym
from .._test import TestCase


class TestSingleComplexGrassmannManifold(TestCase):
    def setUp(self):
        self.m = m = 5
        self.n = n = 2
        self.k = k = 1
        self.man = ComplexGrassmann(m, n, k=k)

        self.proj = lambda x, u: u - np.dot(x, np.dot(x.T, u))

    def test_inner(self):
        X = self.man.rand()
        G = self.man.randvec(X)
        H = self.man.randvec(X)
        np_testing.assert_almost_equal(np.real(np.trace(np.conjugate(G.T)@H)),
                                       self.man.inner(X, G, H))
        assert np.isreal(self.man.inner(X, G, H))

    def test_proj(self):
        # Test proj(proj(X)) == proj(X)
        # and proj(X) belongs to the horizontal space of Stiefel
        X = self.man.rand()
        U = rnd.randn(self.m, self.n) + 1j*rnd.randn(self.m, self.n)
        proj_U = self.man.proj(X, U)
        proj_proj_U = self.man.proj(X, proj_U)

        np_testing.assert_allclose(proj_U, proj_proj_U)

        np_testing.assert_allclose(multiprod(multihconj(X), proj_U),
                                   np.zeros((self.n, self.n)), atol=1e-10)

    def test_norm(self):
        X = self.man.rand()
        U = self.man.randvec(X)
        np_testing.assert_almost_equal(np.trace(np.conjugate(U.T)@U),
                                       self.man.norm(X, U))
        assert np.isreal(self.man.norm(X, U))

    def test_rand(self):
        # Just make sure that things generated are on the manifold
        # and that if you generate two they are not equal.
        # Test also that matrices are complex.
        X = self.man.rand()
        np_testing.assert_allclose(multiprod(multihconj(X), X),
                                   np.eye(self.n), atol=1e-10)
        Y = self.man.rand()
        assert la.norm(X - Y) > 1e-6
        assert np.iscomplex(X).all()

    def test_randvec(self):
        # Just make sure that things generated are on the horizontal space of
        # complex Stiefel manifold
        # and that if you generate two they are not equal.
        # Test also that matrices are complex.
        X = self.man.rand()
        G = self.man.randvec(X)
        np_testing.assert_allclose(multiprod(multihconj(X), G),
                                   np.zeros((self.n, self.n)), atol=1e-10)
        H = self.man.randvec(X)
        assert la.norm(G - H) > 1e-6
        assert np.iscomplex(G).all()

    def test_dist(self):
        X = self.man.rand()
        Y = self.man.rand()
        np_testing.assert_almost_equal(self.man.norm(X, self.man.log(X, Y)),
                                       self.man.dist(X, Y))

    def test_exp_log_inverse(self):
        X = self.man.rand()
        Y = self.man.rand()
        U = self.man.log(X, Y)
        Z = self.man.exp(X, U)
        np_testing.assert_almost_equal(0, self.man.dist(Y, Z), decimal=5)

    def test_log_exp_inverse(self):
        X = self.man.rand()
        U = self.man.randvec(X)
        Y = self.man.exp(X, U)
        V = self.man.log(X, Y)
        # Check that the manifold difference between the tangent vectors u and
        # v is 0
        np_testing.assert_almost_equal(0, self.man.norm(X, U - V))

    def test_retr(self):
        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        x = self.man.rand()
        u = self.man.randvec(x)

        xretru = self.man.retr(x, u)

        np_testing.assert_allclose(multiprod(multihconj(xretru), xretru),
                                   np.eye(self.n),
                                   atol=1e-10)

        u = u * 1e-6
        xretru = self.man.retr(x, u)
        np_testing.assert_allclose(xretru, x + u)


class TestMultiComplexGrassmannManifold(TestCase):
    def setUp(self):
        self.m = m = 5
        self.n = n = 2
        self.k = k = 3
        self.man = ComplexGrassmann(m, n, k=k)

    def test_dim(self):
        assert self.man.dim == self.k * 2 * (self.m * self.n - self.n ** 2)

    def test_typicaldist(self):
        np_testing.assert_almost_equal(self.man.typicaldist,
                                       np.sqrt(self.n * self.k))

    def test_inner(self):
        X = self.man.rand()
        G = self.man.randvec(X)
        H = self.man.randvec(X)
        np_testing.assert_allclose(
            np.real(np.sum(np.conjugate(G) * H)),
            self.man.inner(X, G, H))
        assert np.isreal(self.man.inner(X, G, H))

    def test_proj(self):
        # Test proj(proj(X)) == proj(X) and proj(X)
        # belongs to the horizontal space of Stiefel
        X = self.man.rand()
        U = (rnd.randn(self.k, self.m, self.n)
             + 1j*rnd.randn(self.k, self.m, self.n))
        proj_U = self.man.proj(X, U)
        proj_proj_U = self.man.proj(X, proj_U)

        np_testing.assert_allclose(proj_U, proj_proj_U)

        np_testing.assert_allclose(multiprod(multihconj(X), proj_U),
                                   np.zeros((self.k, self.n, self.n)),
                                   atol=1e-10)

    def test_norm(self):
        X = self.man.rand()
        U = self.man.randvec(X)
        np_testing.assert_almost_equal(self.man.norm(X, U), la.norm(U))
        assert np.isreal(self.man.norm(X, U))

    def test_rand(self):
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        X = self.man.rand()
        np_testing.assert_allclose(multiprod(multihconj(X), X),
                                   multieye(self.k, self.n), atol=1e-10)
        Y = self.man.rand()
        assert la.norm(X - Y) > 1e-6
        assert np.iscomplex(X).all()

    def test_randvec(self):
        # Make sure things generated are in tangent space and if you generate
        # two then they are not equal.
        X = self.man.rand()
        U = self.man.randvec(X)
        np_testing.assert_allclose(multisym(multiprod(multihconj(X), U)),
                                   np.zeros((self.k, self.n, self.n)),
                                   atol=1e-10)
        V = self.man.randvec(X)
        assert la.norm(U - V) > 1e-6
        assert np.iscomplex(U).all()

    def test_dist(self):
        X = self.man.rand()
        Y = self.man.rand()
        np_testing.assert_almost_equal(self.man.dist(X, Y),
                                       self.man.norm(X, self.man.log(X, Y)))

    def test_exp_log_inverse(self):
        X = self.man.rand()
        Y = self.man.rand()
        U = self.man.log(X, Y)
        Z = self.man.exp(X, U)
        np_testing.assert_almost_equal(0, self.man.dist(Y, Z), decimal=5)

    def test_log_exp_inverse(self):
        X = self.man.rand()
        U = self.man.randvec(X)
        Y = self.man.exp(X, U)
        V = self.man.log(X, Y)
        # Check that the manifold difference between the tangent vectors u and
        # v is 0
        np_testing.assert_almost_equal(0, self.man.norm(X, U - V))

    def test_retr(self):
        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        x = self.man.rand()
        u = self.man.randvec(x)

        xretru = self.man.retr(x, u)

        np_testing.assert_allclose(multiprod(multihconj(xretru), xretru),
                                   multieye(self.k, self.n),
                                   atol=1e-10)

        u = u * 1e-6
        xretru = self.man.retr(x, u)
        np_testing.assert_allclose(xretru, x + u)
