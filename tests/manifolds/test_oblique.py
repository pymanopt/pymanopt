import pytest

from pymanopt.manifolds import Oblique
from pymanopt.numerics import NumpyNumericsBackend


class TestObliqueManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.m = m = 100
        self.n = n = 50
        self.backend = NumpyNumericsBackend()
        self.manifold = Oblique(m, n, backend=self.backend)

    # def test_dim(self):

    # def test_typical_dist(self):

    # def test_dist(self):

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
        x = s.random_point()
        y = s.random_point()
        u = s.log(x, y)
        z = s.exp(x, u)
        self.backend.assert_almost_equal(0, s.dist(y, z))

    def test_log_exp_inverse(self):
        s = self.manifold
        x = s.random_point()
        u = s.random_tangent_vector(x)
        y = s.exp(x, u)
        v = s.log(x, y)
        # Check that the manifold difference between the tangent vectors u and
        # v is 0
        self.backend.assert_almost_equal(0, s.norm(x, u - v))

    def test_pair_mean(self):
        s = self.manifold
        X = s.random_point()
        Y = s.random_point()
        Z = s.pair_mean(X, Y)
        self.backend.assert_array_almost_equal(s.dist(X, Z), s.dist(Y, Z))
