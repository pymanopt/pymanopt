import autograd.numpy as np
import pytest
from numpy import testing as np_testing

from pymanopt.manifolds import GeneralizedStiefel


class TestGeneralizedStiefelManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.m = m = 20
        self.n = n = 2
        A = np.random.rand(m, m)
        B = np.dot(A, A.transpose())
        self.B = B
        self.manifold = GeneralizedStiefel(m, n, B)
        self.manifold_polar = GeneralizedStiefel(m, n, B, retraction="polar")
        self.projection = (
            lambda x, u: u - x @ (x.T @ self.B @ u + u.T @ self.B.T @ x) / 2
        )
        self.sym = lambda A: 0.5 * (A + A.T)

    def test_dim(self):
        assert self.manifold.dim == 0.5 * self.n * (2 * self.m - self.n - 1)

    def test_inner_product(self):
        X = self.manifold.random_point()
        A, B = np.random.normal(size=(2, self.m, self.n))
        np_testing.assert_allclose(
            np.trace(A.T @ self.B @ B), self.manifold.inner_product(X, A, B)
        )

    def test_projection(self):
        # Construct a random point X on the manifold.
        X = self.manifold.random_point()
        # Construct a vector H in the ambient space.
        H = np.random.normal(size=(self.m, self.n))
        # Compare the projections.
        Hproj = H - X @ (X.T @ self.B @ H + H.T @ self.B.T @ X) / 2
        np_testing.assert_allclose(Hproj, self.manifold.projection(X, H))

    @pytest.mark.parametrize(
        "manifold_attribute", ["manifold", "manifold_polar"]
    )
    def test_random_point(self, manifold_attribute):
        manifold = getattr(self, manifold_attribute)
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        X = manifold.random_point()
        np_testing.assert_allclose(
            X.T @ self.B @ X, np.eye(self.n), atol=1e-10
        )
        Y = manifold.random_point()
        assert np.linalg.norm(X - Y) > 1e-6

    def test_random_tangent_vector(self):
        # Make sure things generated are in tangent space and if you generate
        # two then they are not equal.
        X = self.manifold.random_point()
        U = self.manifold.random_tangent_vector(X)
        np_testing.assert_allclose(
            self.sym(X.T @ self.B @ U), np.zeros((self.n, self.n)), atol=1e-10
        )
        V = self.manifold.random_tangent_vector(X)
        assert np.linalg.norm(U - V) > 1e-6

    @pytest.mark.parametrize(
        "manifold_attribute", ["manifold", "manifold_polar"]
    )
    def test_retraction(self, manifold_attribute):
        manifold = getattr(self, manifold_attribute)

        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        x = manifold.random_point()
        u = manifold.random_tangent_vector(x)
        xretru = manifold.retraction(x, u)
        np_testing.assert_allclose(
            xretru.T @ self.B @ xretru, np.eye(self.n), atol=1e-10
        )
        u = u * 1e-6
        xretru = manifold.retraction(x, u)
        np_testing.assert_allclose(xretru, x + u)

    def test_norm(self):
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)
        np_testing.assert_almost_equal(
            self.manifold.norm(x, u),
            np.sqrt(np.sum(np.trace(u.T @ self.B @ u, axis1=-2, axis2=-1))),
        )
