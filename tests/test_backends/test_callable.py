import numpy as np
from numpy import random as rnd, testing as np_testing

from pymanopt.function import Callable
from .._test import TestCase


class TestCallableBackend(TestCase):
    def setUp(self):
        self.n = 10

        @Callable
        def nary_cost(x, y):
            return np.sum(x * y)

        self.cost = self.nary_cost = nary_cost

        @Callable
        def nested_nary_cost(x, y, z):
            return np.sum(x ** 2 * y + 3 * z)

        self.nested_nary_cost = nested_nary_cost

    def test_nary_cost_function(self):
        n = self.n
        x, y = [rnd.randn(n) for _ in range(2)]
        np_testing.assert_allclose(np.sum(x * y), self.nary_cost(x, y))

    def test_nested_nary_cost_function(self):
        n = self.n
        x, y, z = [rnd.randn(n) for _ in range(3)]
        np_testing.assert_allclose(np.sum(x ** 2 * y + 3 * z),
                                   self.nested_nary_cost(x, y, z))

    def test_gradient_hessian_exceptions(self):
        with self.assertRaises(NotImplementedError):
            self.cost.compute_gradient()
        with self.assertRaises(NotImplementedError):
            self.cost.compute_hessian_vector_product()
