import pytest

from pymanopt.manifolds import Elliptope


class TestElliptopeManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        n = 50
        k = 10
        self.manifold = Elliptope(n, k)

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

    # def test_exp_log_inverse(self):
    # s = self.manifold
    # X = s.random_point()
    # U = s.random_tangent_vector(X)
    # Uexplog = s.exp(X, s.log(X, U))
    # np_testing.assert_array_almost_equal(U, Uexplog)

    # def test_log_exp_inverse(self):
    # s = self.manifold
    # X = s.random_point()
    # U = s.random_tangent_vector(X)
    # Ulogexp = s.log(X, s.exp(X, U))
    # np_testing.assert_array_almost_equal(U, Ulogexp)

    # def test_pair_mean(self):
    # s = self.manifold
    # X = s.random_point()
    # Y = s.random_point()
    # Z = s.pair_mean(X, Y)
    # np_testing.assert_array_almost_equal(s.dist(X, Z), s.dist(Y, Z))
