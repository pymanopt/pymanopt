import pytest

from pymanopt.manifolds import Grassmann
from pymanopt.numerics import NumpyNumericsBackend
from pymanopt.tools import testing


class TestSingleGrassmannManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.m = m = 5
        self.n = n = 2
        self.k = k = 1
        self.backend = NumpyNumericsBackend()
        self.manifold = Grassmann(m, n, k=k, backend=self.backend)

        self.projection = lambda x, u: u - x @ x.T @ u

    def test_dist(self):
        x = self.manifold.random_point()
        y = self.manifold.random_point()
        self.backend.assert_almost_equal(
            self.manifold.dist(x, y),
            self.manifold.norm(x, self.manifold.log(x, y)),
        )

    def test_euclidean_to_riemannian_hessian(self):
        # Test this function at some randomly generated point.
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)
        egrad = self.backend.random_normal(size=(self.m, self.n))
        ehess = self.backend.random_normal(size=(self.m, self.n))

        self.backend.assert_allclose(
            testing.euclidean_to_riemannian_hessian(self.projection)(
                x, egrad, ehess, u
            ),
            self.manifold.euclidean_to_riemannian_hessian(x, egrad, ehess, u),
        )

    def test_retraction(self):
        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)

        xretru = self.manifold.retraction(x, u)

        self.backend.assert_allclose(
            self.backend.transpose(xretru) @ xretru,
            self.backend.eye(self.n),
            atol=1e-10,
        )

        u = u * 1e-6
        xretru = self.manifold.retraction(x, u)
        self.backend.assert_allclose(xretru, x + u)

    # def test_norm(self):

    def test_random_point(self):
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        X = self.manifold.random_point()
        self.backend.assert_allclose(
            self.backend.transpose(X) @ X, self.backend.eye(self.n), atol=1e-10
        )
        Y = self.manifold.random_point()
        assert self.backend.linalg_norm(X - Y) > 1e-6

    # def test_random_tangent_vector(self):

    # def test_transport(self):

    def test_exp_log_inverse(self):
        s = self.manifold
        x = s.random_point()
        y = s.random_point()
        u = s.log(x, y)
        z = s.exp(x, u)
        self.backend.assert_almost_equal(0, self.manifold.dist(y, z))

    def test_log_exp_inverse(self):
        s = self.manifold
        x = s.random_point()
        u = s.random_tangent_vector(x)
        y = s.exp(x, u)
        v = s.log(x, y)
        # Check that the manifold difference between the tangent vectors u and
        # v is 0
        self.backend.assert_almost_equal(0, self.manifold.norm(x, u - v))

    # def test_pair_mean(self):
    # s = self.manifold
    # X = s.random_point()
    # Y = s.random_point()
    # Z = s.pair_mean(X, Y)
    # self.backend.assert_array_almost_equal(s.dist(X, Z), s.dist(Y, Z))


class TestMultiGrassmannManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.m = m = 5
        self.n = n = 2
        self.k = k = 3
        self.backend = NumpyNumericsBackend()
        self.manifold = Grassmann(m, n, k=k, backend=self.backend)

        self.projection = lambda x, u: u - x @ x.T @ u

    def test_dim(self):
        assert self.manifold.dim == self.k * (self.m * self.n - self.n**2)

    def test_typical_dist(self):
        self.backend.assert_almost_equal(
            self.manifold.typical_dist, self.backend.sqrt(self.n * self.k)
        )

    def test_dist(self):
        x = self.manifold.random_point()
        y = self.manifold.random_point()
        self.backend.assert_almost_equal(
            self.manifold.dist(x, y),
            self.manifold.norm(x, self.manifold.log(x, y)),
        )

    def test_inner_product(self):
        X = self.manifold.random_point()
        A = self.manifold.random_tangent_vector(X)
        B = self.manifold.random_tangent_vector(X)
        self.backend.assert_allclose(
            self.backend.sum(A * B), self.manifold.inner_product(X, A, B)
        )

    def test_projection(self):
        # Construct a random point X on the manifold.
        X = self.manifold.random_point()

        # Construct a vector H in the ambient space.
        H = self.backend.random_normal(size=(self.k, self.m, self.n))

        # Compare the projections.
        Hproj = H - X @ self.backend.transpose(X) @ H
        self.backend.assert_allclose(Hproj, self.manifold.projection(X, H))

    def test_retraction(self):
        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)

        xretru = self.manifold.retraction(x, u)

        self.backend.assert_allclose(
            self.backend.transpose(xretru) @ xretru,
            self.backend.multieye(self.k, self.n),
            atol=1e-10,
        )

        u = u * 1e-6
        xretru = self.manifold.retraction(x, u)
        self.backend.assert_allclose(xretru, x + u)

    def test_norm(self):
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)
        self.backend.assert_almost_equal(
            self.manifold.norm(x, u), self.backend.linalg_norm(u)
        )

    def test_random_point(self):
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        X = self.manifold.random_point()
        self.backend.assert_allclose(
            self.backend.transpose(X) @ X,
            self.backend.multieye(self.k, self.n),
            atol=1e-10,
        )
        Y = self.manifold.random_point()
        assert self.backend.linalg_norm(X - Y) > 1e-6

    def test_random_tangent_vector(self):
        # Make sure things generated are in tangent space and if you generate
        # two then they are not equal.
        X = self.manifold.random_point()
        U = self.manifold.random_tangent_vector(X)
        self.backend.assert_allclose(
            self.backend.sym(self.backend.transpose(X) @ U),
            self.backend.zeros((self.k, self.n, self.n)),
            atol=1e-10,
        )
        V = self.manifold.random_tangent_vector(X)
        assert self.backend.linalg_norm(U - V) > 1e-6

    # def test_transport(self):

    def test_exp_log_inverse(self):
        s = self.manifold
        x = s.random_point()
        y = s.random_point()
        u = s.log(x, y)
        z = s.exp(x, u)
        self.backend.assert_almost_equal(0, self.manifold.dist(y, z))

    def test_log_exp_inverse(self):
        s = self.manifold
        x = s.random_point()
        u = s.random_tangent_vector(x)
        y = s.exp(x, u)
        v = s.log(x, y)
        # Check that the manifold difference between the tangent vectors u and
        # v is 0
        self.backend.assert_almost_equal(0, self.manifold.norm(x, u - v))

    # def test_pair_mean(self):
    # s = self.manifold
    # X = s.random_point()
    # Y = s.random_point()
    # Z = s.pair_mean(X, Y)
    # self.backend.assert_array_almost_equal(s.dist(X, Z), s.dist(Y, Z))
