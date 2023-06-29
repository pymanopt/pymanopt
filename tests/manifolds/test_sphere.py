import warnings

import autograd.numpy as np
import pytest
from numpy import testing as np_testing

import pymanopt
from pymanopt.manifolds import (
    Sphere,
    SphereSubspaceComplementIntersection,
    SphereSubspaceIntersection,
)
from pymanopt.tools import testing


class TestSphereManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.m = m = 100
        self.n = n = 50
        self.manifold = Sphere(m, n)

        # For automatic testing of euclidean_to_riemannian_hessian
        self.projection = lambda x, u: u - np.tensordot(x, u, np.ndim(u)) * x

    def test_dim(self):
        assert self.manifold.dim == self.m * self.n - 1

    def test_typical_dist(self):
        np_testing.assert_almost_equal(self.manifold.typical_dist, np.pi)

    def test_dist(self):
        s = self.manifold
        x = s.random_point()
        y = s.random_point()
        correct_dist = np.arccos(np.tensordot(x, y))
        np_testing.assert_almost_equal(correct_dist, s.dist(x, y))

    def test_inner_product(self):
        s = self.manifold
        x = s.random_point()
        u = s.random_tangent_vector(x)
        v = s.random_tangent_vector(x)
        np_testing.assert_almost_equal(np.sum(u * v), s.inner_product(x, u, v))

    def test_projection(self):
        #  Construct a random point X on the manifold.
        X = np.random.normal(size=(self.m, self.n))
        X /= np.linalg.norm(X, "fro")

        #  Construct a vector H in the ambient space.
        H = np.random.normal(size=(self.m, self.n))

        #  Compare the projections.
        np_testing.assert_array_almost_equal(
            H - X * np.trace(X.T @ H), self.manifold.projection(X, H)
        )

    def test_euclidean_to_riemannian_gradient(self):
        # Should be the same as proj
        #  Construct a random point X on the manifold.
        X = np.random.normal(size=(self.m, self.n))
        X /= np.linalg.norm(X, "fro")

        #  Construct a vector H in the ambient space.
        H = np.random.normal(size=(self.m, self.n))

        #  Compare the projections.
        np_testing.assert_array_almost_equal(
            H - X * np.trace(X.T @ H),
            self.manifold.euclidean_to_riemannian_gradient(X, H),
        )

    def test_euclidean_to_riemannian_hessian(self):
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)
        egrad = np.random.normal(size=(self.m, self.n))
        ehess = np.random.normal(size=(self.m, self.n))

        np_testing.assert_allclose(
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
        np_testing.assert_almost_equal(np.linalg.norm(xretru), 1)

        u = u * 1e-6
        xretru = self.manifold.retraction(x, u)
        np_testing.assert_allclose(xretru, x + u)

    def test_norm(self):
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)

        np_testing.assert_almost_equal(
            self.manifold.norm(x, u), np.linalg.norm(u)
        )

    def test_random_point(self):
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        s = self.manifold
        x = s.random_point()
        np_testing.assert_almost_equal(np.linalg.norm(x), 1)
        y = s.random_point()
        assert np.linalg.norm(x - y) > 1e-3

    def test_random_tangent_vector(self):
        # Just make sure that things generated are in the tangent space and
        # that if you generate two they are not equal.
        s = self.manifold
        x = s.random_point()
        u = s.random_tangent_vector(x)
        v = s.random_tangent_vector(x)
        np_testing.assert_almost_equal(np.tensordot(x, u), 0)

        assert np.linalg.norm(u - v) > 1e-3

    def test_transport(self):
        # Should be the same as proj
        s = self.manifold
        x = s.random_point()
        y = s.random_point()
        u = s.random_tangent_vector(x)

        np_testing.assert_allclose(s.transport(x, y, u), s.projection(y, u))

    def test_exp_log_inverse(self):
        s = self.manifold
        X = s.random_point()
        Y = s.random_point()
        Yexplog = s.exp(X, s.log(X, Y))
        np_testing.assert_array_almost_equal(Y, Yexplog)

    def test_log_exp_inverse(self):
        s = self.manifold
        X = s.random_point()
        U = s.random_tangent_vector(X)
        Ulogexp = s.log(X, s.exp(X, U))
        np_testing.assert_array_almost_equal(U, Ulogexp)

    def test_pair_mean(self):
        s = self.manifold
        X = s.random_point()
        Y = s.random_point()
        Z = s.pair_mean(X, Y)
        np_testing.assert_array_almost_equal(s.dist(X, Z), s.dist(Y, Z))


class TestSphereSubspaceIntersectionManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.n = 2
        # Defines the 1-sphere intersected with the 1-dimensional subspace
        # passing through (1, 1) / sqrt(2). This creates a 0-dimensional
        # manifold as it only consists of isolated points in R^2.
        self.U = np.ones((self.n, 1)) / np.sqrt(2)
        with warnings.catch_warnings(record=True):
            self.manifold = SphereSubspaceIntersection(self.U)

    def test_dim(self):
        assert self.manifold.dim == 0

    def test_random_point(self):
        x = self.manifold.random_point()
        p = np.ones(2) / np.sqrt(2)
        assert np.allclose(x, p) or np.allclose(x, -p)

    def test_projection(self):
        h = np.random.normal(size=self.n)
        x = self.manifold.random_point()
        p = self.manifold.projection(x, h)
        # Since the manifold is 0-dimensional, the tangent at each point is
        # simply the 0-dimensional space {0}.
        np_testing.assert_array_almost_equal(p, np.zeros(self.n))

    def test_dim_1(self):
        U = np.zeros((3, 2))
        U[0, 0] = U[1, 1] = 1
        manifold = SphereSubspaceIntersection(U)
        # U spans the x-y plane, therefore the manifold consists of the
        # 1-sphere in the x-y plane, and has dimension 1.
        assert manifold.dim == 1
        # Check if a random element from the manifold has vanishing
        # z-component.
        x = manifold.random_point()
        np_testing.assert_almost_equal(x[-1], 0)

    def test_dim_rand(self):
        n = 100
        U = np.random.normal(size=(n, n // 3))
        dim = np.linalg.matrix_rank(U) - 1
        manifold = SphereSubspaceIntersection(U)
        assert manifold.dim == dim


class TestSphereSubspaceIntersectionManifoldGradient:
    @pytest.fixture(autouse=True)
    def setup(self):
        span_matrix = pymanopt.manifolds.Stiefel(73, 37).random_point()
        self.manifold = SphereSubspaceIntersection(span_matrix)


class TestSphereSubspaceComplementIntersectionManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.n = 2
        # Define the 1-sphere intersected with the 1-dimensional subspace
        # orthogonal to the line passing through (1, 1) / sqrt(2). This creates
        # a 0-dimensional manifold as it only consits of isolated points in
        # R^2.
        self.U = np.ones((self.n, 1)) / np.sqrt(2)
        with warnings.catch_warnings(record=True):
            self.manifold = SphereSubspaceComplementIntersection(self.U)

    def test_dim(self):
        assert self.manifold.dim == 0

    def test_random_point(self):
        x = self.manifold.random_point()
        p = np.array([-1, 1]) / np.sqrt(2)
        assert np.allclose(x, p) or np.allclose(x, -p)

    def test_projection(self):
        h = np.random.normal(size=self.n)
        x = self.manifold.random_point()
        p = self.manifold.projection(x, h)
        # Since the manifold is 0-dimensional, the tangent at each point is
        # simply the 0-dimensional space {0}.
        np_testing.assert_array_almost_equal(p, np.zeros(self.n))

    def test_dim_1(self):
        U = np.zeros((3, 1))
        U[-1, -1] = 1
        manifold = SphereSubspaceComplementIntersection(U)
        # U spans the z-axis with its orthogonal complement being the x-y
        # plane, therefore the manifold consists of the 1-sphere in the x-y
        # plane, and has dimension 1.
        assert manifold.dim == 1
        # Check if a random element from the manifold has vanishing
        # z-component.
        x = manifold.random_point()
        np_testing.assert_almost_equal(x[-1], 0)

    def test_dim_rand(self):
        n = 100
        U = np.random.normal(size=(n, n // 3))
        # By the rank-nullity theorem the orthogonal complement of span(U) has
        # dimension n - rank(U).
        dim = n - np.linalg.matrix_rank(U) - 1
        manifold = SphereSubspaceComplementIntersection(U)
        assert manifold.dim == dim

        # Test if a random element really lies in the left null space of U.
        x = manifold.random_point()
        np_testing.assert_almost_equal(np.linalg.norm(x), 1)
        np_testing.assert_array_almost_equal(U.T @ x, np.zeros(U.shape[1]))


class TestSphereSubspaceComplementIntersectionManifoldGradient:
    @pytest.fixture(autouse=True)
    def setup(self):
        span_matrix = pymanopt.manifolds.Stiefel(73, 37).random_point()
        self.manifold = SphereSubspaceComplementIntersection(span_matrix)
