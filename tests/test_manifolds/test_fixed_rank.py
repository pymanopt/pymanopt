import numpy as np
from numpy import linalg as la, testing as np_testing

from pymanopt.manifolds import FixedRankEmbedded
from .._test import TestCase


class TestFixedRankEmbeddedManifold(TestCase):
    def setUp(self):
        self.m = m = 10
        self.n = n = 5
        self.k = k = 3
        self.man = FixedRankEmbedded(m, n, k)

    def test_dim(self):
        assert self.man.dim == (self.m + self.n - self.k) * self.k

    def test_typicaldist(self):
        assert self.man.dim == self.man.typicaldist

    def test_dist(self):
        e = self.man
        a = e.rand()
        x = e.randvec(a)
        y = e.randvec(a)
        with self.assertRaises(NotImplementedError):
            e.dist(x, y)

    def test_inner(self):
        e = self.man
        x = e.rand()
        a = e.randvec(x)
        b = e.randvec(x)
        # First embed in the ambient space
        A = x[0].dot(a[1].dot(x[2])) + a[0].dot(x[2]) + x[0].dot(a[2].T)
        B = x[0].dot(b[1].dot(x[2])) + b[0].dot(x[2]) + x[0].dot(b[2].T)
        trueinner = np.sum(A*B)
        np_testing.assert_almost_equal(trueinner, e.inner(x, a, b))

    def test_proj_range(self):
        m = self.man
        x = m.rand()
        v = np.random.randn(self.m, self.n)

        g = m.proj(x, v)
        # Check that g is a true tangent vector
        np_testing.assert_allclose(np.dot(g[0].T, x[0]),
                                   np.zeros((self.k, self.k)),
                                   atol=1e-6)
        np_testing.assert_allclose(np.dot(g[2].T, x[2].T),
                                   np.zeros((self.k, self.k)),
                                   atol=1e-6)

    def test_proj(self):
        # Verify that proj gives the closest point within the tangent space
        # by displacing the result slightly and checking that this increases
        # the distance.
        m = self.man
        x = self.man.rand()
        v = np.random.randn(self.m, self.n)

        g = m.proj(x, v)
        # Displace g a little
        g_disp = g + 0.01 * m.randvec(x)

        # Return to the ambient representation
        g = m.tangent2ambient(x, g)
        g_disp = m.tangent2ambient(x, g_disp)
        g = g[0].dot(g[1]).dot(g[2].T)
        g_disp = g_disp[0].dot(g_disp[1]).dot(g_disp[2].T)

        assert np.linalg.norm(g - v) < np.linalg.norm(g_disp - v)

    def test_proj_tangents(self):
        # Verify that proj leaves tangent vectors unchanged
        e = self.man
        x = e.rand()
        u = e.randvec(x)
        A = e.proj(x, e.tangent2ambient(x, u))
        B = u
        # diff = [A[k]-B[k] for k in range(len(A))]
        np_testing.assert_allclose(A[0], B[0])
        np_testing.assert_allclose(A[1], B[1])
        np_testing.assert_allclose(A[2], B[2])

    def test_norm(self):
        e = self.man
        x = e.rand()
        u = e.randvec(x)
        np_testing.assert_almost_equal(np.sqrt(e.inner(x, u, u)), e.norm(x, u))

    def test_rand(self):
        e = self.man
        x = e.rand()
        y = e.rand()
        assert np.shape(x[0]) == (self.m, self.k)
        assert np.shape(x[1]) == (self.k,)
        assert np.shape(x[2]) == (self.k, self.n)
        np_testing.assert_allclose(x[0].T.dot(x[0]), np.eye(self.k), atol=1e-6)
        np_testing.assert_allclose(x[2].dot(x[2].T), np.eye(self.k), atol=1e-6)

        assert la.norm(x[0] - y[0]) > 1e-6
        assert la.norm(x[1] - y[1]) > 1e-6
        assert la.norm(x[2] - y[2]) > 1e-6

    def test_transp(self):
        s = self.man
        x = s.rand()
        y = s.rand()
        u = s.randvec(x)
        A = s.transp(x, y, u)
        B = s.proj(y, s.tangent2ambient(x, u))
        diff = [A[k]-B[k] for k in range(len(A))]
        np_testing.assert_almost_equal(s.norm(y, diff), 0)

    def test_apply_ambient(self):
        m = self.man
        z = np.random.randn(self.m, self.n)

        # Set u, s, v so that z = u.dot(s).dot(v.T)
        u, s, v = np.linalg.svd(z, full_matrices=False)
        s = np.diag(s)
        v = v.T

        w = np.random.randn(self.n, self.n)

        np_testing.assert_allclose(z.dot(w), m._apply_ambient(z, w))
        np_testing.assert_allclose(z.dot(w), m._apply_ambient((u, s, v), w))

    def test_apply_ambient_transpose(self):
        m = self.man
        z = np.random.randn(self.n, self.m)

        # Set u, s, v so that z = u.dot(s).dot(v.T)
        u, s, v = np.linalg.svd(z, full_matrices=False)
        s = np.diag(s)
        v = v.T

        w = np.random.randn(self.n, self.n)

        np_testing.assert_allclose(z.T.dot(w),
                                   m._apply_ambient_transpose(z, w))
        np_testing.assert_allclose(z.T.dot(w),
                                   m._apply_ambient_transpose((u, s, v), w))

    def test_tangent2ambient(self):
        m = self.man
        x = m.rand()
        z = m.randvec(x)

        z_ambient = (x[0].dot(z[1]).dot(x[2]) + z[0].dot(x[2]) +
                     x[0].dot(z[2].T))

        u, s, v = m.tangent2ambient(x, z)

        np_testing.assert_allclose(z_ambient, u.dot(s).dot(v.T))

    def test_ehess2rhess(self):
        pass

    def test_retr(self):
        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        x = self.man.rand()
        u = self.man.randvec(x)

        y = self.man.retr(x, u)

        np_testing.assert_allclose(y[0].T.dot(y[0]), np.eye(self.k), atol=1e-6)
        np_testing.assert_allclose(y[2].dot(y[2].T), np.eye(self.k), atol=1e-6)

        u = u * 1e-6
        y = self.man.retr(x, u)
        y = y[0].dot(np.diag(y[1])).dot(y[2])

        u = self.man.tangent2ambient(x, u)
        u = u[0].dot(u[1]).dot(u[2].T)
        x = x[0].dot(np.diag(x[1])).dot(x[2])

        np_testing.assert_allclose(y, x + u, atol=1e-5)

    def test_egrad2rgrad(self):
        # Verify that egrad2rgrad and proj are equivalent.
        m = self.man
        x = m.rand()
        u, s, vt = x

        i = np.eye(self.k)

        f = 1 / (s[..., np.newaxis, :]**2 - s[..., :, np.newaxis]**2 + i)

        du = np.random.randn(self.m, self.k)
        ds = np.random.randn(self.k)
        dvt = np.random.randn(self.k, self.n)

        Up = (np.dot(np.dot(np.eye(self.m) - np.dot(u, u.T), du),
                     np.linalg.inv(np.diag(s))))
        M = (np.dot(f * (np.dot(u.T, du) - np.dot(du.T, u)), np.diag(s)) +
             np.dot(np.diag(s), f * (np.dot(vt, dvt.T) - np.dot(dvt, vt.T))) +
             np.diag(ds))
        Vp = (np.dot(np.dot(np.eye(self.n) - np.dot(vt.T, vt), dvt.T),
                     np.linalg.inv(np.diag(s))))

        up, m, vp = m.egrad2rgrad(x, (du, ds, dvt))

        np_testing.assert_allclose(Up, up)
        np_testing.assert_allclose(M, m)
        np_testing.assert_allclose(Vp, vp)

    def test_randvec(self):
        e = self.man
        x = e.rand()
        u = e.randvec(x)

        # Check that u is a tangent vector
        assert np.shape(u[0]) == (self.m, self.k)
        assert np.shape(u[1]) == (self.k, self.k)
        assert np.shape(u[2]) == (self.n, self.k)
        np_testing.assert_allclose(np.dot(u[0].T, x[0]),
                                   np.zeros((self.k, self.k)),
                                   atol=1e-6)
        np_testing.assert_allclose(np.dot(u[2].T, x[2].T),
                                   np.zeros((self.k, self.k)),
                                   atol=1e-6)

        v = e.randvec(x)

        np_testing.assert_almost_equal(e.norm(x, u), 1)
        assert e.norm(x, u - v) > 1e-6
