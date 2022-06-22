import numpy.testing as np_testing

from pymanopt.manifolds import Oblique

from ._manifold_tests import ManifoldTestCase


class TestObliqueManifold(ManifoldTestCase):
    def setUp(self):
        self.m = m = 100
        self.n = n = 50
        self.manifold = Oblique(m, n)

        super().setUp()

    # def test_dim(self):

    # def test_typical_dist(self):

    # def test_dist(self):

    # def test_inner_product(self):

    # def test_projection(self):

    # def test_euclidean_to_riemannian_hessian(self):

    # def test_retraction(self):

    def test_first_order_function_approximation(self):
        self.run_gradient_approximation_test()

    def test_second_order_function_approximation(self):
        self.run_hessian_approximation_test()

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
        np_testing.assert_almost_equal(0, s.dist(y, z), decimal=6)

    def test_log_exp_inverse(self):
        s = self.manifold
        x = s.random_point()
        u = s.random_tangent_vector(x)
        y = s.exp(x, u)
        v = s.log(x, y)
        # Check that the manifold difference between the tangent vectors u and
        # v is 0
        np_testing.assert_almost_equal(0, s.norm(x, u - v))

    def test_pair_mean(self):
        s = self.manifold
        X = s.random_point()
        Y = s.random_point()
        Z = s.pair_mean(X, Y)
        np_testing.assert_array_almost_equal(s.dist(X, Z), s.dist(Y, Z))
