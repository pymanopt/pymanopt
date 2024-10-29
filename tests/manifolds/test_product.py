import pytest

import pymanopt
from pymanopt.backends.numpy_backend import NumpyBackend
from pymanopt.manifolds import Euclidean, Grassmann, Product, Sphere


# TODO: test setting backends


class TestProductManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.m = m = 100
        self.n = n = 50
        self.backend = NumpyBackend()
        self.euclidean = Euclidean(m, n, backend=self.backend)
        self.sphere = Sphere(n, backend=self.backend)
        self.manifold = Product([self.euclidean, self.sphere])

        point = self.manifold.random_point()

        @pymanopt.function.autograd(self.manifold)
        def cost(*x):
            return self.backend.sum(
                [
                    self.backend.linalg_norm(a - b) ** 2
                    for a, b in zip(x, point)
                ]
            )

        self.cost = cost

    def test_dim(self):
        self.backend.assert_equal(
            self.manifold.dim, self.m * self.n + self.n - 1
        )

    def test_typical_dist(self):
        self.backend.assert_equal(
            self.manifold.typical_dist,
            self.backend.sqrt((self.m * self.n) + self.backend.pi**2),
        )

    def test_dist(self):
        X = self.manifold.random_point()
        Y = self.manifold.random_point()
        self.backend.assert_equal(
            self.manifold.dist(X, Y),
            self.backend.sqrt(
                self.euclidean.dist(X[0], Y[0]) ** 2
                + self.sphere.dist(X[1], Y[1]) ** 2
            ),
        )

    def test_tangent_vector_multiplication(self):
        # Regression test for https://github.com/pymanopt/pymanopt/issues/49.
        manifold = Product(
            (
                Euclidean(12, backend=self.backend),
                Grassmann(12, 3, backend=self.backend),
            )
        )
        x = manifold.random_point()
        eta = manifold.random_tangent_vector(x)
        factor = 42
        eta_scaled = eta * factor
        eta_euclidean, eta_grassmann = eta
        eta_euclidean_scaled, eta_grassmann_scaled = eta_scaled
        assert self.backend.allclose(
            eta_euclidean * factor, eta_euclidean_scaled
        )
        assert self.backend.allclose(
            eta_grassmann * factor, eta_grassmann_scaled
        )

    # def test_inner_product(self):

    # def test_projection(self):

    # def test_euclidean_to_riemannian_hessian(self):

    # def test_retraction(self):

    # def test_norm(self):

    # def test_random_point(self):

    # def test_random_tangent_vector(self):

    # def test_transport(self):

    def test_exp_log_inverse(self):
        s = self.manifold
        X = s.random_point()
        Y = s.random_point()
        Yexplog = s.exp(X, tangent_vector=s.log(X, Y))
        self.backend.assert_allclose(s.dist(point_a=Y, point_b=Yexplog), 0)

    def test_log_exp_inverse(self):
        s = self.manifold
        X = s.random_point()
        U = s.random_tangent_vector(X)
        Ulogexp = s.log(X, s.exp(X, U))
        self.backend.assert_allclose(U[0], Ulogexp[0])
        self.backend.assert_allclose(U[1], Ulogexp[1])

    def test_pair_mean(self):
        s = self.manifold
        X = s.random_point()
        Y = s.random_point()
        Z = s.pair_mean(X, Y)
        self.backend.assert_allclose(s.dist(X, Z), s.dist(Y, Z))
