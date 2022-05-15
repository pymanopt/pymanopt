import numpy.testing as np_testing

from pymanopt.manifolds import SpecialOrthogonalGroup
from pymanopt.tools.multi import multiprod, multitransp

from .._test import TestCase


class TestSpecialOrthogonalGroup(TestCase):
    def setUp(self):
        self.n = n = 10
        self.k = k = 3
        self.so_product = SpecialOrthogonalGroup(n, k=k)
        self.so = SpecialOrthogonalGroup(n)

    def test_random_point(self):
        point = self.so.random_point()
        assert point.shape == (self.n, self.n)
        np_testing.assert_almost_equal(point.T @ point - point @ point.T, 0)

        point = self.so_product.random_point()
        assert point.shape == (self.k, self.n, self.n)
        np_testing.assert_almost_equal(
            multiprod(multitransp(point), point)
            - multiprod(point, multitransp(point)),
            0,
        )

    def test_random_tangent_vector(self):
        point = self.so.random_point()
        tangent_vector = self.so.random_tangent_vector(point)
        np_testing.assert_almost_equal(tangent_vector, -tangent_vector.T)

        point = self.so_product.random_point()
        tangent_vector = self.so_product.random_tangent_vector(point)
        np_testing.assert_almost_equal(
            tangent_vector, -multitransp(tangent_vector)
        )

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
