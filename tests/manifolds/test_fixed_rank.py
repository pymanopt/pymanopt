import pytest

from pymanopt.backends import Backend
from pymanopt.manifolds import FixedRankEmbedded


class TestFixedRankEmbeddedManifold:
    @pytest.fixture(autouse=True)
    def setup(self, real_numerics_backend: Backend):
        self.m = m = 10
        self.n = n = 5
        self.k = k = 3
        self.backend = real_numerics_backend
        self.manifold = FixedRankEmbedded(m, n, k, backend=self.backend)

        # u, s, vt = self.manifold.random_point()
        # matrix = (u * s) @ vt
        # @pymanopt.function.autograd(self.manifold)
        # def cost(u, s, vt):
        #     return bk.linalg_norm((u * s) @ vt - matrix) ** 2
        # self.cost = cost

    def test_dim(self):
        assert self.manifold.dim == (self.m + self.n - self.k) * self.k

    def test_typical_dist(self):
        assert self.manifold.dim == self.manifold.typical_dist

    def test_dist(self):
        e = self.manifold
        a = e.random_point()
        x = e.random_tangent_vector(a)
        y = e.random_tangent_vector(a)
        with pytest.raises(NotImplementedError):
            e.dist(x, y)

    def test_inner_product(self):
        bk = self.backend
        e = self.manifold
        x = e.random_point()
        a = e.random_tangent_vector(x)
        b = e.random_tangent_vector(x)
        # First embed in the ambient space
        A = x[0] @ a[1] @ x[2] + a[0] @ x[2] + x[0] @ a[2].T
        B = x[0] @ b[1] @ x[2] + b[0] @ x[2] + x[0] @ b[2].T
        bk.assert_allclose(bk.sum(A * B), e.inner_product(x, a, b))

    def test_proj_range(self):
        bk = self.backend
        m = self.manifold
        x = m.random_point()
        v = bk.random_normal(size=(self.m, self.n))

        g = m.projection(x, v)
        # Check that g is a true tangent vector
        bk.assert_allclose(
            g[0].T @ x[0], bk.zeros((self.k, self.k)), atol=1e-6
        )
        bk.assert_allclose(
            g[2].T @ x[2].T, bk.zeros((self.k, self.k)), atol=1e-6
        )

    def test_projection(self):
        bk = self.backend
        # Verify that proj gives the closest point within the tangent space
        # by displacing the result slightly and checking that this increases
        # the distance.
        m = self.manifold
        x = self.manifold.random_point()
        v = bk.random_normal(size=(self.m, self.n))

        g = m.projection(x, v)
        # Displace g a little
        g_disp = g + 0.01 * m.random_tangent_vector(x)

        # Return to the ambient representation
        g = m.embedding(x, g)
        g_disp = m.embedding(x, g_disp)
        g = g[0] @ g[1] @ g[2].T
        g_disp = g_disp[0] @ g_disp[1] @ g_disp[2].T

        assert bk.linalg_norm(g - v) < bk.linalg_norm(g_disp - v)

    def test_proj_tangents(self):
        bk = self.backend
        # Verify that proj leaves tangent vectors unchanged
        e = self.manifold
        x = e.random_point()
        u = e.random_tangent_vector(x)
        A = e.projection(x, e.embedding(x, u))
        B = u
        # diff = [A[k]-B[k] for k in range(len(A))]
        bk.assert_allclose(A[0], B[0])
        bk.assert_allclose(A[1], B[1])
        bk.assert_allclose(A[2], B[2])

    def test_norm(self):
        bk = self.backend
        e = self.manifold
        x = e.random_point()
        u = e.random_tangent_vector(x)
        bk.assert_almost_equal(bk.sqrt(e.inner_product(x, u, u)), e.norm(x, u))

    def test_random_point(self):
        bk = self.backend
        e = self.manifold
        x = e.random_point()
        assert x[0].shape == (self.m, self.k)
        assert x[1].shape == (self.k,)
        assert x[2].shape == (self.k, self.n)
        bk.assert_allclose(x[0].T @ x[0], bk.eye(self.k), atol=1e-6)
        bk.assert_allclose(x[2] @ x[2].T, bk.eye(self.k), atol=1e-6)

        y = e.random_point()
        assert bk.linalg_norm(x[0] - y[0]) > 1e-6
        assert bk.linalg_norm(x[1] - y[1]) > 1e-6
        assert bk.linalg_norm(x[2] - y[2]) > 1e-6

    def test_transport(self):
        bk = self.backend
        s = self.manifold
        x = s.random_point()
        y = s.random_point()
        u = s.random_tangent_vector(x)
        A = s.transport(x, y, u)
        B = s.projection(y, s.embedding(x, u))
        diff = [A[k] - B[k] for k in range(len(A))]
        bk.assert_almost_equal(s.norm(y, diff), 0)

    def test_apply_ambient(self):
        bk = self.backend
        m = self.manifold
        z = bk.random_normal(size=(self.m, self.n))

        # Set u, s, v so that z = u @ s @ v.T
        u, s, v = bk.linalg_svd(z, full_matrices=False)
        s = bk.diag(s)
        v = v.T

        w = bk.random_normal(size=(self.n, self.n))

        bk.assert_allclose(z @ w, m._apply_ambient(z, w))
        bk.assert_allclose(z @ w, m._apply_ambient((u, s, v), w), atol=1e-5)

    def test_apply_ambient_transpose(self):
        bk = self.backend
        m = self.manifold
        z = bk.random_normal(size=(self.n, self.m))

        # Set u, s, v so that z = u @ s @ v.T
        u, s, v = bk.linalg_svd(z, full_matrices=False)
        s = bk.diag(s)
        v = v.T

        w = bk.random_normal(size=(self.n, self.n))

        bk.assert_allclose(z.T @ w, m._apply_ambient_transpose(z, w))
        bk.assert_allclose(z.T @ w, m._apply_ambient_transpose((u, s, v), w))

    def test_embedding(self):
        bk = self.backend
        m = self.manifold
        x = m.random_point()
        z = m.random_tangent_vector(x)

        z_ambient = x[0] @ z[1] @ x[2] + z[0] @ x[2] + x[0] @ z[2].T

        u, s, v = m.embedding(x, z)

        bk.assert_allclose(z_ambient, u @ s @ v.T)

    def test_euclidean_to_riemannian_hessian(self):
        pass

    def test_retraction(self):
        bk = self.backend
        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)

        y = self.manifold.retraction(x, u)

        bk.assert_allclose(y[0].T @ y[0], bk.eye(self.k), atol=1e-6)
        bk.assert_allclose(y[2] @ y[2].T, bk.eye(self.k), atol=1e-6)

        u = u * 1e-6
        y = self.manifold.retraction(x, u)
        y = y[0] @ bk.diag(y[1]) @ y[2]

        u = self.manifold.embedding(x, u)
        u = u[0] @ u[1] @ u[2].T
        x = x[0] @ bk.diag(x[1]) @ x[2]

        bk.assert_allclose(y, x + u, atol=1e-5)

    def test_euclidean_to_riemannian_gradient(self):
        bk = self.backend
        # Verify that euclidean_to_riemannian_gradient and proj are equivalent.
        m = self.manifold
        x = m.random_point()
        u, s, vt = x

        i = bk.eye(self.k)

        f = 1 / (s[..., bk.newaxis, :] ** 2 - s[..., :, bk.newaxis] ** 2 + i)

        du = bk.random_normal(size=(self.m, self.k))
        ds = bk.random_normal(size=self.k)
        dvt = bk.random_normal(size=(self.k, self.n))

        Up = (bk.eye(self.m) - u @ u.T) @ du @ bk.linalg_inv(bk.diag(s))
        M = (
            f * (u.T @ du - du.T @ u) @ bk.diag(s)
            + bk.diag(s) @ f * (vt @ dvt.T - dvt @ vt.T)
            + bk.diag(ds)
        )
        Vp = (bk.eye(self.n) - vt.T @ vt) @ dvt.T @ bk.linalg_inv(bk.diag(s))

        up, m, vp = m.euclidean_to_riemannian_gradient(x, (du, ds, dvt))

        bk.assert_allclose(Up, up)
        bk.assert_allclose(M, m)
        bk.assert_allclose(Vp, vp)

    def test_random_tangent_vector(self):
        bk = self.backend
        e = self.manifold
        x = e.random_point()
        u = e.random_tangent_vector(x)

        # Check that u is a tangent vector
        assert u[0].shape == (self.m, self.k)
        assert u[1].shape == (self.k, self.k)
        assert u[2].shape == (self.n, self.k)
        bk.assert_allclose(
            u[0].T @ x[0], bk.zeros((self.k, self.k)), atol=1e-6
        )
        bk.assert_allclose(
            u[2].T @ x[2].T, bk.zeros((self.k, self.k)), atol=1e-6
        )

        v = e.random_tangent_vector(x)

        bk.assert_almost_equal(e.norm(x, u), 1)
        assert e.norm(x, u - v) > 1e-6
