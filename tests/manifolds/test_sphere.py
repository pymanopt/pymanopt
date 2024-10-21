import warnings

import pytest

from pymanopt.manifolds import (
    Sphere,
    SphereSubspaceComplementIntersection,
    SphereSubspaceIntersection,
)


class TestSphereManifold:
    @pytest.fixture(
        autouse=True,
    )
    def setup(self, real_numerics_backend):
        self.m = m = 100
        self.n = n = 50
        self.backend = real_numerics_backend
        self.manifold = Sphere(m, n, backend=self.backend)

        # For automatic testing of euclidean_to_riemannian_hessian
        self.projection = (
            lambda x, u: u
            - self.backend.tensordot(x, u, self.backend.ndim(u)) * x
        )

    def test_dim(self):
        assert self.manifold.dim == self.m * self.n - 1

    def test_typical_dist(self):
        self.backend.assert_allclose(
            self.manifold.typical_dist, self.backend.pi
        )

    def test_dist(self):
        s = self.manifold
        x = s.random_point()
        y = s.random_point()
        correct_dist = self.backend.arccos(self.backend.tensordot(x, y))
        self.backend.assert_allclose(correct_dist, s.dist(x, y))

    def test_inner_product(self):
        s = self.manifold
        x = s.random_point()
        u = s.random_tangent_vector(x)
        v = s.random_tangent_vector(x)
        self.backend.assert_allclose(
            self.backend.sum(u * v),
            s.inner_product(x, u, v),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_projection(self):
        #  Construct a random point X on the manifold.
        X = self.backend.random_normal(size=(self.m, self.n))
        X /= self.backend.linalg_norm(X, "fro")

        #  Construct a vector H in the ambient space.
        H = self.backend.random_normal(size=(self.m, self.n))

        #  Compare the projections.
        self.backend.assert_allclose(
            H - X * self.backend.trace(self.backend.transpose(X) @ H),
            self.manifold.projection(X, H),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_euclidean_to_riemannian_gradient(self):
        # Should be the same as proj
        #  Construct a random point X on the manifold.
        X = self.backend.random_normal(size=(self.m, self.n))
        X /= self.backend.linalg_norm(X, "fro")

        #  Construct a vector H in the ambient space.
        H = self.backend.random_normal(size=(self.m, self.n))

        #  Compare the projections.
        self.backend.assert_allclose(
            H - X * self.backend.trace(self.backend.transpose(X) @ H),
            self.manifold.euclidean_to_riemannian_gradient(X, H),
            rtol=1e-6,
            atol=1e-6,
        )

    # def test_euclidean_to_riemannian_hessian(self):
    #     x = self.manifold.random_point()
    #     u = self.manifold.random_tangent_vector(x)
    #     egrad = self.backend.random_normal(size=(self.m, self.n))
    #     ehess = self.backend.random_normal(size=(self.m, self.n))
    #
    #     # TODO: testing.euclidean_to_riemannian_hessian() was only implemented
    #     # for numpy arrays, we have to implement a generic version for all
    #     # backends. This will probably need more sync between numerics and
    #     # autodiff backend (one single class?).
    #     self.backend.assert_allclose(
    #         testing.euclidean_to_riemannian_hessian(self.projection)(
    #             x, egrad, ehess, u
    #         ),
    #         self.manifold.euclidean_to_riemannian_hessian(x, egrad, ehess, u),
    #     )

    def test_retraction(self):
        # Test that the result is on the manifold and that for small
        # tangent vectors it has little effect.
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)

        xretru = self.manifold.retraction(x, u)
        self.backend.assert_allclose(self.backend.linalg_norm(xretru), 1)

        u = u * 1e-6
        xretru = self.manifold.retraction(x, u)
        self.backend.assert_allclose(xretru, x + u, rtol=1e-6, atol=1e-6)

    def test_norm(self):
        x = self.manifold.random_point()
        u = self.manifold.random_tangent_vector(x)

        self.backend.assert_allclose(
            self.manifold.norm(x, u), self.backend.linalg_norm(u)
        )

    def test_random_point(self):
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        s = self.manifold
        x = s.random_point()
        self.backend.assert_allclose(
            self.backend.linalg_norm(x), 1, rtol=1e-6, atol=1e-6
        )
        y = s.random_point()
        assert self.backend.linalg_norm(x - y) > 1e-3

    def test_random_tangent_vector(self):
        # Just make sure that things generated are in the tangent space and
        # that if you generate two they are not equal.
        s = self.manifold
        x = s.random_point()
        u = s.random_tangent_vector(x)
        v = s.random_tangent_vector(x)
        self.backend.assert_allclose(
            self.backend.tensordot(x, u), 0, rtol=1e-6, atol=1e-6
        )

        assert self.backend.linalg_norm(u - v) > 1e-3

    def test_transport(self):
        # Should be the same as proj
        s = self.manifold
        x = s.random_point()
        y = s.random_point()
        u = s.random_tangent_vector(x)

        self.backend.assert_allclose(s.transport(x, y, u), s.projection(y, u))

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
        self.backend.assert_allclose(
            s.dist(X, Z), s.dist(Y, Z), rtol=1e-6, atol=1e-6
        )


class TestSphereSubspaceIntersectionManifold:
    @pytest.fixture(autouse=True)
    def setup(self, real_numerics_backend):
        self.n = 2
        self.backend = real_numerics_backend
        # Defines the 1-sphere intersected with the 1-dimensional subspace
        # passing through (1, 1) / sqrt(2). This creates a 0-dimensional
        # manifold as it only consists of isolated points in R^2.
        self.U = self.backend.ones((self.n, 1)) / self.backend.sqrt(2)
        with warnings.catch_warnings(record=True):
            self.manifold = SphereSubspaceIntersection(
                self.U, backend=self.backend
            )

    def test_dim(self):
        assert self.manifold.dim == 0

    def test_random_point(self):
        x = self.manifold.random_point()
        p = self.backend.ones(2) / self.backend.sqrt(2)
        assert self.backend.allclose(x, p) or self.backend.allclose(x, -p)

    def test_projection(self):
        h = self.backend.random_normal(size=self.n)
        x = self.manifold.random_point()
        p = self.manifold.projection(x, h)
        # Since the manifold is 0-dimensional, the tangent at each point is
        # simply the 0-dimensional space {0}.
        self.backend.assert_allclose(
            p, self.backend.zeros(self.n), rtol=1e-6, atol=1e-6
        )

    def test_dim_1(self):
        U = self.backend.vstack(
            (self.backend.eye(2), self.backend.zeros((1, 2)))
        )
        manifold = SphereSubspaceIntersection(U, backend=self.backend)
        # U spans the x-y plane, therefore the manifold consists of the
        # 1-sphere in the x-y plane, and has dimension 1.
        assert manifold.dim == 1
        # Check if a random element from the manifold has vanishing
        # z-component.
        x = manifold.random_point()
        self.backend.assert_allclose(x[-1], 0)

    def test_dim_rand(self):
        n = 100
        U = self.backend.random_normal(size=(n, n // 3))
        dim = self.backend.linalg_matrix_rank(U) - 1
        manifold = SphereSubspaceIntersection(U, backend=self.backend)
        assert manifold.dim == dim


# class TestSphereSubspaceIntersectionManifoldGradient:
#     @pytest.fixture(autouse=True)
#     def setup(self):
#         span_matrix = pymanopt.manifolds.Stiefel(73, 37).random_point()
#         self.manifold = SphereSubspaceIntersection(
#             span_matrix, backend=NumpyNumericsBackend()
#         )


class TestSphereSubspaceComplementIntersectionManifold:
    @pytest.fixture(autouse=True)
    def setup(self, real_numerics_backend):
        self.n = 2
        self.backend = real_numerics_backend
        # Define the 1-sphere intersected with the 1-dimensional subspace
        # orthogonal to the line passing through (1, 1) / sqrt(2). This creates
        # a 0-dimensional manifold as it only consits of isolated points in
        # R^2.
        self.U = self.backend.ones((self.n, 1)) / self.backend.sqrt(2)
        with warnings.catch_warnings(record=True):
            self.manifold = SphereSubspaceComplementIntersection(
                self.U, backend=self.backend
            )

    def test_dim(self):
        assert self.manifold.dim == 0

    def test_random_point(self):
        x = self.manifold.random_point()
        p = self.backend.array([-1, 1]) / self.backend.sqrt(2)
        assert self.backend.allclose(
            x, p, atol=1e-6, rtol=1e-6
        ) or self.backend.allclose(x, -p, atol=1e-6, rtol=1e-6)

    def test_projection(self):
        h = self.backend.random_normal(size=self.n)
        x = self.manifold.random_point()
        p = self.manifold.projection(x, h)
        # Since the manifold is 0-dimensional, the tangent at each point is
        # simply the 0-dimensional space {0}.
        self.backend.assert_allclose(
            p, self.backend.zeros(self.n), atol=1e-5, rtol=1e-6
        )

    def test_dim_1(self):
        U = self.backend.array([[0.0], [0.0], [1.0]])
        manifold = SphereSubspaceComplementIntersection(
            U, backend=self.backend
        )
        # U spans the z-axis with its orthogonal complement being the x-y
        # plane, therefore the manifold consists of the 1-sphere in the x-y
        # plane, and has dimension 1.
        assert manifold.dim == 1
        # Check if a random element from the manifold has vanishing
        # z-component.
        x = manifold.random_point()
        self.backend.assert_allclose(x[-1], 0)

    def test_dim_rand(self):
        n = 100
        U = self.backend.random_normal(size=(n, n // 3))
        # By the rank-nullity theorem the orthogonal complement of span(U) has
        # dimension n - rank(U).
        dim = n - self.backend.linalg_matrix_rank(U) - 1
        manifold = SphereSubspaceComplementIntersection(
            U, backend=self.backend
        )
        assert manifold.dim == dim

        # Test if a random element really lies in the left null space of U.
        x = manifold.random_point()
        self.backend.assert_allclose(self.backend.linalg_norm(x), 1)
        self.backend.assert_allclose(
            self.backend.squeeze(
                self.backend.transpose(U) @ self.backend.reshape(x, (-1, 1))
            ),
            self.backend.zeros(U.shape[1]),
            atol=1.5e-6,
            rtol=1e-6,
        )


# class TestSphereSubspaceComplementIntersectionManifoldGradient:
#     @pytest.fixture(autouse=True)
#     def setup(self):
#         span_matrix = pymanopt.manifolds.Stiefel(73, 37).random_point()
#         self.manifold = SphereSubspaceComplementIntersection(
#             span_matrix, backend=NumpyNumericsBackend()
#         )
