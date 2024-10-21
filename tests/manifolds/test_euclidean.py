import pytest

from pymanopt.manifolds import ComplexEuclidean, Euclidean


class TestEuclideanManifold:
    @pytest.fixture(autouse=True)
    def setup(self, real_numerics_backend):
        self.m = m = 10
        self.n = n = 5
        self.backend = real_numerics_backend
        self.manifold = Euclidean(m, n, backend=self.backend)

    def test_dim(self):
        assert self.manifold.dim == self.m * self.n

    def test_typical_dist(self):
        self.backend.assert_allclose(
            self.manifold.typical_dist, self.backend.sqrt(self.m * self.n)
        )

    def test_dist(self):
        e = self.manifold
        x, y = self.backend.random_normal(size=(2, self.m, self.n))
        self.backend.assert_allclose(
            e.dist(x, y), self.backend.linalg_norm(x - y)
        )

    def test_inner_product(self):
        e = self.manifold
        x = e.random_point()
        y = e.random_tangent_vector(x)
        z = e.random_tangent_vector(x)
        self.backend.assert_allclose(
            self.backend.real(self.backend.sum(self.backend.conjugate(y) * z)),
            e.inner_product(x, y, z),
        )

    def test_projection(self):
        e = self.manifold
        x = e.random_point()
        u = e.random_tangent_vector(x)
        self.backend.assert_allclose(e.projection(x, u), u)

    def test_euclidean_to_riemannian_hessian(self):
        e = self.manifold
        x = e.random_point()
        u = e.random_tangent_vector(x)
        egrad, ehess = self.backend.random_normal(size=(2, self.m, self.n))
        self.backend.assert_allclose(
            e.euclidean_to_riemannian_hessian(x, egrad, ehess, u), ehess
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
        u = self.backend.random_normal(size=(self.m, self.n))
        self.backend.assert_allclose(self.backend.linalg_norm(u), e.norm(x, u))

    def test_random_point(self):
        e = self.manifold
        x = e.random_point()
        y = e.random_point()
        assert x.shape == (self.m, self.n)
        assert self.backend.linalg_norm(x - y) > 1e-6

    def test_random_tangent_vector(self):
        e = self.manifold
        x = e.random_point()
        u = e.random_tangent_vector(x)
        v = e.random_tangent_vector(x)
        assert u.shape == (self.m, self.n)
        self.backend.assert_allclose(self.backend.linalg_norm(u), 1)
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
        self.backend.assert_allclose(Y, Yexplog)

    def test_log_exp_inverse(self):
        s = self.manifold
        X = s.random_point()
        U = s.random_tangent_vector(X)
        Ulogexp = s.log(X, s.exp(X, U))
        self.backend.assert_allclose(U, Ulogexp)

    def test_pair_mean(self):
        s = self.manifold
        X = s.random_point()
        Y = s.random_point()
        Z = s.pair_mean(X, Y)
        self.backend.assert_allclose(s.dist(X, Z), s.dist(Y, Z))


class TestComplexEuclideanManifold(TestEuclideanManifold):
    @pytest.fixture(autouse=True)
    def setup(self, complex_numerics_backend):
        self.m = m = 10
        self.n = n = 5
        self.backend = complex_numerics_backend
        self.manifold = ComplexEuclidean(m, n, backend=self.backend)

    def test_dim(self):
        assert self.manifold.dim == 2 * self.m * self.n

    def test_typical_dist(self):
        nx = self.backend
        nx.assert_allclose(
            self.manifold.typical_dist, self.backend.sqrt(2 * self.m * self.n)
        )

    def test_random_point(self):
        e = self.manifold
        x = e.random_point()
        y = e.random_point()
        assert x.dtype == self.backend.dtype
        assert x.shape == (self.m, self.n)
        # assert self.backend.linalg_norm(x - y) > 1e-6
        assert not self.backend.allclose(x, y)

    def test_random_tangent_vector(self):
        e = self.manifold
        x = e.random_point()
        u = e.random_tangent_vector(x)
        v = e.random_tangent_vector(x)
        assert u.dtype == self.backend.dtype
        assert u.shape == (self.m, self.n)
        self.backend.assert_allclose(self.backend.linalg_norm(u), 1)
        assert not self.backend.allclose(u, v)
        # assert self.backend.linalg_norm(u - v) > 1e-6
