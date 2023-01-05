import autograd.numpy as np
import pytest
from numpy import testing as np_testing

from pymanopt.manifolds import ComplexGrassmann
from pymanopt.tools.multi import multieye, multihconj, multisym


class TestSingleComplexGrassmannManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.m = m = 5
        self.n = n = 2
        self.k = k = 1
        self.manifold = ComplexGrassmann(m, n, k=k)

        self.projection = lambda x, u: u - x @ x.T @ u

    def test_inner_product(self):
        X = self.manifold.random_point()
        G = self.manifold.random_tangent_vector(X)
        H = self.manifold.random_tangent_vector(X)
        np_testing.assert_almost_equal(
            np.real(np.trace(np.conjugate(G.T) @ H)),
            self.manifold.inner_product(X, G, H),
        )
        assert np.isreal(self.manifold.inner_product(X, G, H))

    def test_projection(self):
        # Test proj(proj(X)) == proj(X)
        # and proj(X) belongs to the horizontal space of Stiefel
        X = self.manifold.random_point()
        U = np.random.normal(size=(self.m, self.n)) + 1j * np.random.normal(
            size=(self.m, self.n)
        )
        proj_U = self.manifold.projection(X, U)
        proj_proj_U = self.manifold.projection(X, proj_U)

        np_testing.assert_allclose(proj_U, proj_proj_U)

        np_testing.assert_allclose(
            multihconj(X) @ proj_U,
            np.zeros((self.n, self.n)),
            atol=1e-10,
        )

    def test_norm(self):
        X = self.manifold.random_point()
        U = self.manifold.random_tangent_vector(X)
        np_testing.assert_almost_equal(
            np.trace(np.conjugate(U.T) @ U), self.manifold.norm(X, U)
        )
        assert np.isreal(self.manifold.norm(X, U))

    def test_random_point(self):
        # Just make sure that things generated are on the manifold
        # and that if you generate two they are not equal.
        # Test also that matrices are complex.
        X = self.manifold.random_point()
        np_testing.assert_allclose(
            multihconj(X) @ X, np.eye(self.n), atol=1e-10
        )
        Y = self.manifold.random_point()
        assert np.linalg.norm(X - Y) > 1e-6
        assert np.iscomplex(X).all()

    def test_random_tangent_vector(self):
        # Just make sure that things generated are on the horizontal space of
        # complex Stiefel manifold
        # and that if you generate two they are not equal.
        # Test also that matrices are complex.
        X = self.manifold.random_point()
        G = self.manifold.random_tangent_vector(X)
        np_testing.assert_allclose(
            multihconj(X) @ G, np.zeros((self.n, self.n)), atol=1e-10
        )
        H = self.manifold.random_tangent_vector(X)
        assert np.linalg.norm(G - H) > 1e-6
        assert np.iscomplex(G).all()

    def test_dist(self):
        X = self.manifold.random_point()
        Y = self.manifold.random_point()
        np_testing.assert_almost_equal(
            self.manifold.norm(X, self.manifold.log(X, Y)),
            self.manifold.dist(X, Y),
        )

    def test_exp_log_inverse(self):
        X = self.manifold.random_point()
        Y = self.manifold.random_point()
        U = self.manifold.log(X, Y)
        Z = self.manifold.exp(X, U)
        np_testing.assert_almost_equal(0, self.manifold.dist(Y, Z), decimal=5)

    def test_log_exp_inverse(self):
        X = self.manifold.random_point()
        U = self.manifold.random_tangent_vector(X)
        Y = self.manifold.exp(X, U)
        V = self.manifold.log(X, Y)
        # Check that the manifold difference between the tangent vectors u and
        # v is 0
        np_testing.assert_almost_equal(0, self.manifold.norm(X, U - V))

    def test_retraction(self):
        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)

        xretru = self.manifold.retraction(x, u)

        np_testing.assert_allclose(
            multihconj(xretru) @ xretru, np.eye(self.n), atol=1e-10
        )

        u = u * 1e-6
        xretru = self.manifold.retraction(x, u)
        np_testing.assert_allclose(xretru, x + u)


class TestMultiComplexGrassmannManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.m = m = 5
        self.n = n = 2
        self.k = k = 3
        self.manifold = ComplexGrassmann(m, n, k=k)

    def test_dim(self):
        assert self.manifold.dim == self.k * 2 * (
            self.m * self.n - self.n**2
        )

    def test_typical_dist(self):
        np_testing.assert_almost_equal(
            self.manifold.typical_dist, np.sqrt(self.n * self.k)
        )

    def test_inner_product(self):
        X = self.manifold.random_point()
        G = self.manifold.random_tangent_vector(X)
        H = self.manifold.random_tangent_vector(X)
        np_testing.assert_allclose(
            np.real(np.sum(np.conjugate(G) * H)),
            self.manifold.inner_product(X, G, H),
        )
        assert np.isreal(self.manifold.inner_product(X, G, H))

    def test_projection(self):
        # Test proj(proj(X)) == proj(X) and proj(X)
        # belongs to the horizontal space of Stiefel
        X = self.manifold.random_point()
        U = np.random.normal(
            size=(self.k, self.m, self.n)
        ) + 1j * np.random.normal(size=(self.k, self.m, self.n))
        proj_U = self.manifold.projection(X, U)
        proj_proj_U = self.manifold.projection(X, proj_U)

        np_testing.assert_allclose(proj_U, proj_proj_U)

        np_testing.assert_allclose(
            multihconj(X) @ proj_U,
            np.zeros((self.k, self.n, self.n)),
            atol=1e-10,
        )

    def test_norm(self):
        X = self.manifold.random_point()
        U = self.manifold.random_tangent_vector(X)
        np_testing.assert_almost_equal(
            self.manifold.norm(X, U), np.linalg.norm(U)
        )
        assert np.isreal(self.manifold.norm(X, U))

    def test_random_point(self):
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        X = self.manifold.random_point()
        np_testing.assert_allclose(
            multihconj(X) @ X, multieye(self.k, self.n), atol=1e-10
        )
        Y = self.manifold.random_point()
        assert np.linalg.norm(X - Y) > 1e-6
        assert np.iscomplex(X).all()

    def test_random_tangent_vector(self):
        # Make sure things generated are in tangent space and if you generate
        # two then they are not equal.
        X = self.manifold.random_point()
        U = self.manifold.random_tangent_vector(X)
        np_testing.assert_allclose(
            multisym(multihconj(X) @ U),
            np.zeros((self.k, self.n, self.n)),
            atol=1e-10,
        )
        V = self.manifold.random_tangent_vector(X)
        assert np.linalg.norm(U - V) > 1e-6
        assert np.iscomplex(U).all()

    def test_dist(self):
        X = self.manifold.random_point()
        Y = self.manifold.random_point()
        np_testing.assert_almost_equal(
            self.manifold.dist(X, Y),
            self.manifold.norm(X, self.manifold.log(X, Y)),
        )

    def test_exp_log_inverse(self):
        X = self.manifold.random_point()
        Y = self.manifold.random_point()
        U = self.manifold.log(X, Y)
        Z = self.manifold.exp(X, U)
        np_testing.assert_almost_equal(0, self.manifold.dist(Y, Z), decimal=5)

    def test_log_exp_inverse(self):
        X = self.manifold.random_point()
        U = self.manifold.random_tangent_vector(X)
        Y = self.manifold.exp(X, U)
        V = self.manifold.log(X, Y)
        # Check that the manifold difference between the tangent vectors u and
        # v is 0
        np_testing.assert_almost_equal(0, self.manifold.norm(X, U - V))

    def test_retraction(self):
        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)

        xretru = self.manifold.retraction(x, u)

        np_testing.assert_allclose(
            multihconj(xretru) @ xretru,
            multieye(self.k, self.n),
            atol=1e-10,
        )

        u = u * 1e-6
        xretru = self.manifold.retraction(x, u)
        np_testing.assert_allclose(xretru, x + u)
