import pytest

from pymanopt.manifolds import ComplexGrassmann, Grassmann
from pymanopt.numerics import NumericsBackend


class TestGrassmannManifold:
    @pytest.fixture(autouse=True)
    def setup(
        self, real_numerics_backend: NumericsBackend, product_dimension: int
    ):
        self.n = n = 5
        self.p = p = 2
        self.k = k = product_dimension
        self.point_shape = (k, n, p) if k > 1 else (n, p)
        self.backend = real_numerics_backend
        self.manifold = Grassmann(n, p, k=k, backend=self.backend)

        self.projection = (
            lambda x, u: u - x @ self.backend.conjugate_transpose(x) @ u
        )

    def test_dim(self):
        assert self.manifold.dim == self.k * (self.n * self.p - self.p**2)

    def test_typical_dist(self):
        self.backend.assert_almost_equal(
            self.manifold.typical_dist, self.backend.sqrt(self.p * self.k)
        )

    def test_dist(self):
        x = self.manifold.random_point()
        y = self.manifold.random_point()
        self.backend.assert_almost_equal(
            self.manifold.dist(x, y),
            self.manifold.norm(x, self.manifold.log(x, y)),
        )

    def test_inner_product(self):
        bk = self.backend
        X = self.manifold.random_point()
        A = self.manifold.random_tangent_vector(X)
        B = self.manifold.random_tangent_vector(X)
        bk.assert_allclose(
            bk.real(bk.tensordot(bk.conjugate(A), B, bk.ndim(A))),
            self.manifold.inner_product(X, A, B),
        )

    def test_projection(self):
        # Construct a random point X on the manifold.
        X = self.manifold.random_point()

        # Construct a vector H in the ambient space.
        H = self.backend.random_normal(size=(self.k, self.n, self.p))

        # Compare the projections.
        Hproj = H - X @ self.backend.conjugate_transpose(X) @ H
        self.backend.assert_allclose(Hproj, self.manifold.projection(X, H))

    def test_retraction(self):
        bk = self.backend
        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)

        xretru = self.manifold.retraction(x, u)

        bk.assert_allclose(
            bk.conjugate_transpose(xretru) @ xretru,
            bk.squeeze(bk.multieye(self.k, self.p)),
        )

        u = u * 1e-6
        xretru = self.manifold.retraction(x, u)
        bk.assert_allclose(xretru, x + u)

    def test_norm(self):
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)
        self.backend.assert_almost_equal(
            self.manifold.norm(x, u), self.backend.linalg_norm(u)
        )

    def test_random_point(self):
        bk = self.backend
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        X = self.manifold.random_point()
        bk.assert_allclose(
            bk.conjugate_transpose(X) @ X,
            bk.squeeze(bk.multieye(self.k, self.p)),
        )
        Y = self.manifold.random_point()
        assert bk.linalg_norm(X - Y) > 1e-6

    def test_random_tangent_vector(self):
        bk = self.backend
        # Make sure things generated are in tangent space and if you generate
        # two then they are not equal.
        X = self.manifold.random_point()
        U = self.manifold.random_tangent_vector(X)
        bk.assert_allclose(
            bk.sym(bk.conjugate_transpose(X) @ U),
            bk.squeeze(bk.zeros((self.k, self.p, self.p))),
        )
        V = self.manifold.random_tangent_vector(X)
        assert bk.linalg_norm(U - V) > 1e-6

    # def test_transport(self):

    def test_exp_log_inverse(self):
        s = self.manifold
        x = s.random_point()
        y = s.random_point()
        u = s.log(x, y)
        z = s.exp(x, u)
        self.backend.assert_allclose(self.manifold.dist(y, z), 0.0, atol=1e-3)

    def test_log_exp_inverse(self):
        s = self.manifold
        x = s.random_point()
        u = s.random_tangent_vector(x)
        y = s.exp(x, u)
        v = s.log(x, y)
        # Check that the manifold difference between the tangent vectors u and
        # v is 0
        self.backend.assert_allclose(0, self.manifold.norm(x, u - v))

    # def test_euclidean_to_riemannian_hessian(self):
    #     # Test this function at some randomly generated point.
    #     x = self.manifold.random_point()
    #     u = self.manifold.random_tangent_vector(x)
    #     egrad = self.backend.random_normal(size=(self.m, self.n))
    #     ehess = self.backend.random_normal(size=(self.m, self.n))

    #     self.backend.assert_allclose(
    #         testing.euclidean_to_riemannian_hessian(self.projection)(
    #             x, egrad, ehess, u
    #         ),
    #         self.manifold.euclidean_to_riemannian_hessian(x, egrad, ehess, u),
    #     )

    # def test_pair_mean(self):
    # s = self.manifold
    # X = s.random_point()
    # Y = s.random_point()
    # Z = s.pair_mean(X, Y)
    # self.backend.assert_array_almost_equal(s.dist(X, Z), s.dist(Y, Z))


class TestComplexGrassmannManifold(TestGrassmannManifold):
    @pytest.fixture(autouse=True)
    def setup(
        self, complex_numerics_backend: NumericsBackend, product_dimension: int
    ):
        self.n = n = 5
        self.p = p = 2
        self.k = k = product_dimension
        self.point_shape = (k, n, p) if k > 1 else (n, p)
        self.backend = complex_numerics_backend
        self.manifold = ComplexGrassmann(n, p, k=k, backend=self.backend)

        self.projection = lambda x, u: u - x @ x.T @ u

    def test_dim(self):
        assert (
            self.manifold.dim == self.k * (self.n * self.p - self.p**2) * 2
        )
