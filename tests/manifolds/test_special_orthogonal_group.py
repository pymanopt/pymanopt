import autograd.numpy as np
import numpy.testing as np_testing
from nose2.tools import params

from pymanopt.manifolds import SpecialOrthogonalGroup
from pymanopt.tools.multi import multieye, multitransp

from ._manifold_tests import ManifoldTestCase


class TestSpecialOrthogonalGroup(ManifoldTestCase):
    def setUp(self):
        self.n = n = 10
        self.k = k = 3
        self.so_product = SpecialOrthogonalGroup(n, k=k)
        self.so = SpecialOrthogonalGroup(n)
        self.so_polar = SpecialOrthogonalGroup(n, retraction="polar")

        self.manifold = self.so
        super().setUp()

    def test_random_point(self):
        point = self.so.random_point()
        assert point.shape == (self.n, self.n)
        np_testing.assert_almost_equal(point.T @ point - point @ point.T, 0)
        np_testing.assert_almost_equal(point.T @ point, np.eye(self.n))
        assert np.allclose(np.linalg.det(point), 1)

        point = self.so_product.random_point()
        assert point.shape == (self.k, self.n, self.n)
        np_testing.assert_almost_equal(
            multitransp(point) @ point - point @ multitransp(point),
            0,
        )
        np_testing.assert_almost_equal(
            multitransp(point) @ point, multieye(self.k, self.n)
        )
        assert np.allclose(np.linalg.det(point), 1)

    def test_random_tangent_vector(self):
        point = self.so.random_point()
        tangent_vector = self.so.random_tangent_vector(point)
        np_testing.assert_almost_equal(tangent_vector, -tangent_vector.T)

        point = self.so_product.random_point()
        tangent_vector = self.so_product.random_tangent_vector(point)
        np_testing.assert_almost_equal(
            tangent_vector, -multitransp(tangent_vector)
        )

    @params("so", "so_polar")
    def test_retraction(self, manifold_attribute):
        manifold = getattr(self, manifold_attribute)

        # Test that the result is on the manifold.
        x = manifold.random_point()
        u = manifold.random_tangent_vector(x)
        xretru = manifold.retraction(x, u)

        np_testing.assert_almost_equal(
            xretru.T @ xretru - xretru @ xretru.T, 0
        )
        np_testing.assert_almost_equal(
            xretru.T @ xretru, np.eye(xretru.shape[-1])
        )
        np_testing.assert_allclose(np.linalg.det(xretru), 1)

    def test_exp_log_inverse(self):
        s = self.so
        X = s.random_point()
        Y = s.random_point()
        Yexplog = s.exp(X, s.log(X, Y))
        np_testing.assert_array_almost_equal(Y, Yexplog)

    def test_log_exp_inverse(self):
        s = self.so
        X = s.random_point()
        U = s.random_tangent_vector(X)
        Ulogexp = s.log(X, s.exp(X, U))
        np_testing.assert_array_almost_equal(U, Ulogexp)

    def test_first_order_function_approximation(self):
        self.run_gradient_approximation_test()

    def test_second_order_function_approximation(self):
        self.run_hessian_approximation_test()
