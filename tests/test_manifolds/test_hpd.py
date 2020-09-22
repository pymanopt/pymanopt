import numpy as np
from numpy import linalg as la, random as rnd, testing as np_testing
from scipy.linalg import eigvalsh

from pymanopt.manifolds import HermitianPositiveDefinite
from pymanopt.tools.multi import multiherm, multiprod, multitransp
from .._test import TestCase


class TestSingleHermitianPositiveDefiniteManifold(TestCase):
    def setUp(self):
        self.n = n = 15
        self.man = HermitianPositiveDefinite(n)

    def test_dim(self):
        man = self.man
        n = self.n
        np_testing.assert_equal(man.dim, n * (n+1))

    def test_rand(self):
        # Just test that rand returns a point on the manifold and two
        # different matrices generated by rand aren't too close together
        n = self.n
        man = self.man
        x = man.rand()

        assert np.shape(x) == (n, n)
        assert x.dtype == np.complex

        # Check symmetry
        np_testing.assert_allclose(x, multiherm(x))

        # Check positivity of eigenvalues
        w = la.eigvalsh(x)
        assert (w > [0]).all()

    def test_randvec(self):
        # Just test that randvec returns an element of the tangent space
        # with norm 1 and that two randvecs are different.
        man = self.man
        x = man.rand()
        u = man.randvec(x)
        v = man.randvec(x)
        np_testing.assert_allclose(multiherm(u), u)
        np_testing.assert_almost_equal(1, man.norm(x, u))
        assert la.norm(u - v) > 1e-3

    def test_inner(self):
        man = self.man
        x = man.rand()
        a = man.randvec(x)
        b = man.randvec(x)
        np.testing.assert_almost_equal(np.real(np.trace(a@b)),
                                       man.inner(x, x@a,
                                                 x@b))
        assert man.inner(x, a, b).dtype == np.float

        x_inv = np.linalg.inv(x)
        inner = np.real(np.trace(x_inv@a@x_inv@b))
        np.testing.assert_almost_equal(inner, man.inner(x, a, b))

    def test_norm(self):
        man = self.man
        u = man.randvec(np.eye(self.n))
        np.testing.assert_almost_equal(man.norm(np.eye(self.n), u), la.norm(u))

        x = man.rand()
        u = man.randvec(x)
        np.testing.assert_almost_equal(
            np.sqrt(man.inner(x, u, u)), man.norm(x, u))

    def test_proj(self):
        man = self.man
        x = man.rand()
        a = rnd.randn(self.n, self.n) + 1j*rnd.randn(self.n, self.n)
        np.testing.assert_allclose(man.proj(x, a), multiherm(a))
        np.testing.assert_allclose(man.proj(x, a), man.proj(x, man.proj(x, a)))

    def test_egrad2rgrad(self):
        man = self.man
        x = man.rand()
        u = rnd.randn(self.n, self.n) + 1j*rnd.randn(self.n, self.n)
        np.testing.assert_allclose(man.egrad2rgrad(x, u),
                                   multiprod(multiprod(x, multiherm(u)), x))

    def test_exp(self):
        # exp(x, u) = x + u.
        man = self.man
        x = man.rand()
        u = man.randvec(x)
        e = man.exp(x, u)

        # Check symmetry
        np_testing.assert_allclose(e, multiherm(e))

        # Check positivity of eigenvalues
        w = la.eigvalsh(e)
        assert (w > [0]).all()

        u = u * 1e-6
        np_testing.assert_allclose(man.exp(x, u), x + u)

    def test_exp_log_inverse(self):
        man = self.man
        x = man.rand()
        y = man.rand()
        u = man.log(x, y)
        np_testing.assert_allclose(man.exp(x, u), y)

    def test_log_exp_inverse(self):
        man = self.man
        x = man.rand()
        u = man.randvec(x)
        y = man.exp(x, u)
        np_testing.assert_allclose(man.log(x, y), u)

    def test_transp(self):
        man = self.man
        x = man.rand()
        y = man.rand()
        u = man.randvec(x)
        np_testing.assert_allclose(man.transp(x, y, u), u)

    def test_dist(self):
        man = self.man
        x = man.rand()
        y = man.rand()

        # Test separability
        np_testing.assert_almost_equal(man.dist(x, x), 0.)

        # Test symmetry
        np_testing.assert_almost_equal(man.dist(x, y), man.dist(y, x))

        # Test alternative implementation
        # from Eq 6.14 of "Positive definite matrices"
        d = np.sqrt((np.log(np.real(eigvalsh(x, y)))**2).sum())
        np_testing.assert_almost_equal(man.dist(x, y), d)

        # check that dist is consistent with log
        np_testing.assert_almost_equal(man.dist(x, y),
                                       man.norm(x, man.log(x, y)))


class TestMultiHermitianPositiveDefiniteManifold(TestCase):
    def setUp(self):
        self.n = n = 10
        self.k = k = 3
        self.man = HermitianPositiveDefinite(n, k)

    def test_dim(self):
        man = self.man
        n = self.n
        k = self.k
        np_testing.assert_equal(man.dim, k * n * (n+1))

    def test_rand(self):
        # Just test that rand returns a point on the manifold and two
        # different matrices generated by rand aren't too close together
        k = self.k
        n = self.n
        man = self.man
        x = man.rand()

        assert np.shape(x) == (k, n, n)
        assert x.dtype == np.complex

        # Check symmetry
        np_testing.assert_allclose(x, multiherm(x))

        # Check positivity of eigenvalues
        w = la.eigvalsh(x)
        assert (w > [[0]]).all()

    def test_randvec(self):
        # Just test that randvec returns an element of the tangent space
        # with norm 1 and that two randvecs are different.
        man = self.man
        x = man.rand()
        u = man.randvec(x)
        v = man.randvec(x)
        np_testing.assert_allclose(multiherm(u), u)
        np_testing.assert_almost_equal(1, man.norm(x, u))
        assert la.norm(u - v) > 1e-3

    def test_inner(self):
        man = self.man
        x = man.rand()
        a = man.randvec(x)
        b = man.randvec(x)
        # b is not symmetric, it is Hermitian
        np.testing.assert_almost_equal(
            np.tensordot(a, multitransp(b), axes=a.ndim),
            man.inner(x, multiprod(x, a),
                      multiprod(x, b)))
        assert man.inner(x, a, b).dtype == np.float

    def test_norm(self):
        man = self.man
        Id = np.array(self.k * [np.eye(self.n)])
        u = man.randvec(Id)
        np.testing.assert_almost_equal(man.norm(Id, u), la.norm(u))

        x = man.rand()
        u = man.randvec(x)
        np.testing.assert_almost_equal(
            np.sqrt(man.inner(x, u, u)), man.norm(x, u))

    def test_proj(self):
        man = self.man
        x = man.rand()
        a = rnd.randn(self.k, self.n, self.n)
        + 1j*rnd.randn(self.k, self.n, self.n)
        np.testing.assert_allclose(man.proj(x, a), multiherm(a))
        np.testing.assert_allclose(man.proj(x, a), man.proj(x, man.proj(x, a)))

    def test_egrad2rgrad(self):
        man = self.man
        x = man.rand()
        u = rnd.randn(self.k, self.n, self.n)
        + 1j*rnd.randn(self.k, self.n, self.n)
        np.testing.assert_allclose(man.egrad2rgrad(x, u),
                                   multiprod(multiprod(x, multiherm(u)), x))

    def test_exp(self):
        # Test against manopt implementation, test that for small vectors
        # exp(x, u) = x + u.
        man = self.man
        x = man.rand()
        u = man.randvec(x)
        e = man.exp(x, u)

        # Check symmetry
        np_testing.assert_allclose(e, multiherm(e))

        # Check positivity of eigenvalues
        w = la.eigvalsh(e)
        assert (w > [[0]]).all()

        u = u * 1e-6
        np_testing.assert_allclose(man.exp(x, u), x + u)

    def test_retr(self):
        # Check that result is on manifold and for small vectors
        # retr(x, u) = x + u.
        man = self.man
        x = man.rand()
        u = man.randvec(x)
        y = man.retr(x, u)

        assert np.shape(y) == (self.k, self.n, self.n)
        # Check symmetry
        np_testing.assert_allclose(y, multiherm(y))

        # Check positivity of eigenvalues
        w = la.eigvalsh(y)
        assert (w > [[0]]).all()

        u = u * 1e-6
        np_testing.assert_allclose(man.retr(x, u), x + u)

    def test_exp_log_inverse(self):
        man = self.man
        x = man.rand()
        y = man.rand()
        u = man.log(x, y)
        np_testing.assert_allclose(man.exp(x, u), y)

    def test_log_exp_inverse(self):
        man = self.man
        x = man.rand()
        u = man.randvec(x)
        y = man.exp(x, u)
        np_testing.assert_allclose(man.log(x, y), u)

    def test_transp(self):
        man = self.man
        x = man.rand()
        y = man.rand()
        u = man.randvec(x)
        np_testing.assert_allclose(man.transp(x, y, u), u)

    def test_dist(self):
        man = self.man
        x = man.rand()
        y = man.rand()

        # Test separability
        np_testing.assert_almost_equal(man.dist(x, x), 0.)

        # Test symmetry
        np_testing.assert_almost_equal(man.dist(x, y), man.dist(y, x))

        # check that dist is consistent with log
        np_testing.assert_almost_equal(man.dist(x, y),
                                       man.norm(x, man.log(x, y)))
