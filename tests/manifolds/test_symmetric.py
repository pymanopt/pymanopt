# import autograd.numpy as self.backend
import pytest

from pymanopt.manifolds import Symmetric


class TestSymmetricManifold:
    @pytest.fixture(autouse=True)
    def setup(self, real_numerics_backend):
        self.n = n = 10
        self.k = k = 5
        self.backend = real_numerics_backend
        self.manifold = Symmetric(n, k, backend=self.backend)

    def test_dim(self):
        assert self.manifold.dim == 0.5 * self.k * self.n * (self.n + 1)

    def test_typical_dist(self):
        manifold = self.manifold
        self.backend.assert_allclose(
            manifold.typical_dist, self.backend.sqrt(manifold.dim)
        )

    def test_dist(self):
        e = self.manifold
        x, y = self.backend.random_normal(size=(2, self.k, self.n, self.n))
        self.backend.assert_allclose(
            e.dist(x, y), self.backend.linalg_norm(x - y)
        )

    def test_inner_product(self):
        e = self.manifold
        x = e.random_point()
        y = e.random_tangent_vector(x)
        z = e.random_tangent_vector(x)
        self.backend.assert_allclose(
            self.backend.sum(y * z),
            e.inner_product(x, y, z),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_projection(self):
        e = self.manifold
        x = e.random_point()
        u = self.backend.random_normal(size=(self.k, self.n, self.n))
        self.backend.assert_allclose(e.projection(x, u), self.backend.sym(u))

    def test_euclidean_to_riemannian_hessian(self):
        e = self.manifold
        x = e.random_point()
        u = e.random_tangent_vector(x)
        egrad, ehess = self.backend.random_normal(
            size=(2, self.k, self.n, self.n)
        )
        self.backend.assert_allclose(
            e.euclidean_to_riemannian_hessian(x, egrad, ehess, u),
            self.backend.sym(ehess),
        )

    def test_retraction(self):
        e = self.manifold
        x = e.random_point()
        u = e.random_tangent_vector(x)
        self.backend.assert_allclose(e.retraction(x, u), x + u)

    def test_euclidean_to_riemannian_gradient(self):
        e = self.manifold
        x = e.random_point()
        u = e.random_tangent_vector(x)
        self.backend.assert_allclose(
            e.euclidean_to_riemannian_gradient(x, u), u
        )

    def test_norm(self):
        e = self.manifold
        x = e.random_point()
        u = self.backend.random_normal(size=(self.n, self.n, self.k))
        self.backend.assert_allclose(
            self.backend.sqrt(self.backend.sum(u**2)), e.norm(x, u)
        )

    def test_random_point(self):
        e = self.manifold
        x = e.random_point()
        y = e.random_point()
        assert x.shape == (self.k, self.n, self.n)
        self.backend.assert_allclose(x, self.backend.sym(x))
        assert self.backend.linalg_norm(x - y) > 1e-6

    def test_random_tangent_vector(self):
        e = self.manifold
        x = e.random_point()
        u = e.random_tangent_vector(x)
        v = e.random_tangent_vector(x)
        assert u.shape == (self.k, self.n, self.n)
        self.backend.assert_allclose(u, self.backend.sym(u))
        self.backend.assert_allclose(
            self.backend.linalg_norm(u), 1, rtol=1e-6, atol=1e-6
        )
        assert self.backend.linalg_norm(u - v) > 1e-6

    def test_transport(self):
        e = self.manifold
        x = e.random_point()
        y = e.random_point()
        u = e.random_tangent_vector(x)
        self.backend.assert_allclose(e.transport(x, y, u), u)

    def test_exp_log_inverse(self):
        s = self.manifold
        X = s.random_point()
        Y = s.random_point()
        Yexplog = s.exp(X, s.log(X, Y))
        self.backend.assert_allclose(Y, Yexplog, rtol=1e-6, atol=1e-6)

    def test_log_exp_inverse(self):
        s = self.manifold
        X = s.random_point()
        U = s.random_tangent_vector(X)
        Ulogexp = s.log(X, s.exp(X, U))
        self.backend.assert_allclose(U, Ulogexp, rtol=1e-6, atol=1e-6)

    def test_pair_mean(self):
        s = self.manifold
        X = s.random_point()
        Y = s.random_point()
        Z = s.pair_mean(X, Y)
        self.backend.assert_allclose(s.dist(X, Z), s.dist(Y, Z))
