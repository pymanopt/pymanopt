# import numpy as np
import pytest
from numpy import complex128

from pymanopt.manifolds import ComplexGrassmann
from pymanopt.numerics import NumpyNumericsBackend


class TestSingleComplexGrassmannManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.m = m = 5
        self.n = n = 2
        self.k = k = 1
        self.backend = NumpyNumericsBackend(dtype=complex128)
        self.manifold = ComplexGrassmann(m, n, k=k, backend=self.backend)

        self.projection = lambda x, u: u - x @ x.T @ u

    def test_inner_product(self):
        X = self.manifold.random_point()
        G = self.manifold.random_tangent_vector(X)
        H = self.manifold.random_tangent_vector(X)
        self.backend.assert_almost_equal(
            self.backend.real(
                self.backend.trace(self.backend.conjugate(G.T) @ H)
            ),
            self.manifold.inner_product(X, G, H),
        )
        assert self.backend.isrealobj(self.manifold.inner_product(X, G, H))

    def test_projection(self):
        # Test proj(proj(X)) == proj(X)
        # and proj(X) belongs to the horizontal space of Stiefel
        X = self.manifold.random_point()
        U = self.backend.random_normal(
            size=(self.m, self.n)
        ) + 1j * self.backend.random_normal(size=(self.m, self.n))
        proj_U = self.manifold.projection(X, U)
        proj_proj_U = self.manifold.projection(X, proj_U)

        self.backend.assert_allclose(proj_U, proj_proj_U)

        self.backend.assert_allclose(
            self.backend.conjugate_transpose(X) @ proj_U,
            self.backend.zeros((self.n, self.n)),
            atol=1e-10,
        )

    def test_norm(self):
        X = self.manifold.random_point()
        U = self.manifold.random_tangent_vector(X)
        self.backend.assert_almost_equal(
            self.backend.trace(self.backend.conjugate(U.T) @ U),
            self.manifold.norm(X, U),
        )
        assert self.backend.isrealobj(self.manifold.norm(X, U))

    def test_random_point(self):
        # Just make sure that things generated are on the manifold
        # and that if you generate two they are not equal.
        # Test also that matrices are complex.
        X = self.manifold.random_point()
        self.backend.assert_allclose(
            self.backend.conjugate_transpose(X) @ X,
            self.backend.eye(self.n),
            atol=1e-10,
        )
        Y = self.manifold.random_point()
        assert self.backend.linalg_norm(X - Y) > 1e-6
        assert self.backend.all(self.backend.iscomplexobj(X))

    def test_random_tangent_vector(self):
        # Just make sure that things generated are on the horizontal space of
        # complex Stiefel manifold
        # and that if you generate two they are not equal.
        # Test also that matrices are complex.
        X = self.manifold.random_point()
        G = self.manifold.random_tangent_vector(X)
        self.backend.assert_allclose(
            self.backend.conjugate_transpose(X) @ G,
            self.backend.zeros((self.n, self.n)),
            atol=1e-10,
        )
        H = self.manifold.random_tangent_vector(X)
        assert self.backend.linalg_norm(G - H) > 1e-6
        assert self.backend.all(self.backend.iscomplexobj(G))

    def test_dist(self):
        X = self.manifold.random_point()
        Y = self.manifold.random_point()
        self.backend.assert_almost_equal(
            self.manifold.norm(X, self.manifold.log(X, Y)),
            self.manifold.dist(X, Y),
        )

    def test_exp_log_inverse(self):
        X = self.manifold.random_point()
        Y = self.manifold.random_point()
        U = self.manifold.log(X, Y)
        Z = self.manifold.exp(X, U)
        self.backend.assert_almost_equal(0, self.manifold.dist(Y, Z))

    def test_log_exp_inverse(self):
        X = self.manifold.random_point()
        U = self.manifold.random_tangent_vector(X)
        Y = self.manifold.exp(X, U)
        V = self.manifold.log(X, Y)
        # Check that the manifold difference between the tangent vectors u and
        # v is 0
        self.backend.assert_almost_equal(0, self.manifold.norm(X, U - V))

    def test_retraction(self):
        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)

        xretru = self.manifold.retraction(x, u)

        self.backend.assert_allclose(
            self.backend.conjugate_transpose(xretru) @ xretru,
            self.backend.eye(self.n),
            atol=1e-10,
        )

        u = u * 1e-6
        xretru = self.manifold.retraction(x, u)
        self.backend.assert_allclose(xretru, x + u)


class TestMultiComplexGrassmannManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.m = m = 5
        self.n = n = 2
        self.k = k = 3
        self.backend = NumpyNumericsBackend(dtype=complex128)
        self.manifold = ComplexGrassmann(m, n, k=k, backend=self.backend)

    def test_dim(self):
        assert self.manifold.dim == self.k * 2 * (
            self.m * self.n - self.n**2
        )

    def test_typical_dist(self):
        self.backend.assert_almost_equal(
            self.manifold.typical_dist, self.backend.sqrt(self.n * self.k)
        )

    def test_inner_product(self):
        X = self.manifold.random_point()
        G = self.manifold.random_tangent_vector(X)
        H = self.manifold.random_tangent_vector(X)
        self.backend.assert_allclose(
            self.backend.real(self.backend.sum(self.backend.conjugate(G) * H)),
            self.manifold.inner_product(X, G, H),
        )
        assert self.backend.isrealobj(self.manifold.inner_product(X, G, H))

    def test_projection(self):
        # Test proj(proj(X)) == proj(X) and proj(X)
        # belongs to the horizontal space of Stiefel
        X = self.manifold.random_point()
        U = self.backend.random_normal(
            size=(self.k, self.m, self.n)
        ) + 1j * self.backend.random_normal(size=(self.k, self.m, self.n))
        proj_U = self.manifold.projection(X, U)
        proj_proj_U = self.manifold.projection(X, proj_U)

        self.backend.assert_allclose(proj_U, proj_proj_U)

        self.backend.assert_allclose(
            self.backend.conjugate_transpose(X) @ proj_U,
            self.backend.zeros((self.k, self.n, self.n)),
            atol=1e-10,
        )

    def test_norm(self):
        X = self.manifold.random_point()
        U = self.manifold.random_tangent_vector(X)
        self.backend.assert_almost_equal(
            self.manifold.norm(X, U), self.backend.linalg_norm(U)
        )
        assert self.backend.isrealobj(self.manifold.norm(X, U))

    def test_random_point(self):
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        X = self.manifold.random_point()
        self.backend.assert_allclose(
            self.backend.conjugate_transpose(X) @ X,
            self.backend.multieye(self.k, self.n),
            atol=1e-10,
        )
        Y = self.manifold.random_point()
        assert self.backend.linalg_norm(X - Y) > 1e-6
        assert self.backend.all(self.backend.iscomplexobj(X))

    def test_random_tangent_vector(self):
        # Make sure things generated are in tangent space and if you generate
        # two then they are not equal.
        X = self.manifold.random_point()
        U = self.manifold.random_tangent_vector(X)
        self.backend.assert_allclose(
            self.backend.sym(self.backend.conjugate_transpose(X) @ U),
            self.backend.zeros((self.k, self.n, self.n)),
            atol=1e-10,
        )
        V = self.manifold.random_tangent_vector(X)
        assert self.backend.linalg_norm(U - V) > 1e-6
        assert self.backend.all(self.backend.iscomplexobj(U))

    def test_dist(self):
        X = self.manifold.random_point()
        Y = self.manifold.random_point()
        self.backend.assert_almost_equal(
            self.manifold.dist(X, Y),
            self.manifold.norm(X, self.manifold.log(X, Y)),
        )

    def test_exp_log_inverse(self):
        X = self.manifold.random_point()
        Y = self.manifold.random_point()
        U = self.manifold.log(X, Y)
        Z = self.manifold.exp(X, U)
        self.backend.assert_almost_equal(0, self.manifold.dist(Y, Z))

    def test_log_exp_inverse(self):
        X = self.manifold.random_point()
        U = self.manifold.random_tangent_vector(X)
        Y = self.manifold.exp(X, U)
        V = self.manifold.log(X, Y)
        # Check that the manifold difference between the tangent vectors u and
        # v is 0
        self.backend.assert_almost_equal(0, self.manifold.norm(X, U - V))

    def test_retraction(self):
        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)

        xretru = self.manifold.retraction(x, u)

        self.backend.assert_allclose(
            self.backend.conjugate_transpose(xretru) @ xretru,
            self.backend.multieye(self.k, self.n),
            atol=1e-10,
        )

        u = u * 1e-6
        xretru = self.manifold.retraction(x, u)
        self.backend.assert_allclose(xretru, x + u)
