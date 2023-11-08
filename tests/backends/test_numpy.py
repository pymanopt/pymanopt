import numpy as np
import pytest
from numpy import testing as np_testing

import pymanopt

from . import _backend_tests


class TestNumPyBackend:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.n = 10

        @pymanopt.function.numpy(
            _backend_tests.manifold_factory(point_layout=3)
        )
        def nary_cost(x, y):
            return np.sum(x * y)

        self.cost = self.nary_cost = nary_cost

        @pymanopt.function.numpy(
            _backend_tests.manifold_factory(point_layout=3)
        )
        def nested_nary_cost(x, y, z):
            return np.sum(x**2 * y + 3 * z)

        self.nested_nary_cost = nested_nary_cost

    def test_nary_cost_function(self):
        n = self.n
        x, y = [np.random.normal(size=n) for _ in range(2)]
        np_testing.assert_allclose(np.sum(x * y), self.nary_cost(x, y))

    def test_nested_nary_cost_function(self):
        n = self.n
        x, y, z = [np.random.normal(size=n) for _ in range(3)]
        np_testing.assert_allclose(
            np.sum(x**2 * y + 3 * z), self.nested_nary_cost(x, y, z)
        )

    def test_gradient_hessian_exceptions(self):
        with pytest.raises(NotImplementedError):
            self.cost.get_gradient_operator()
        with pytest.raises(NotImplementedError):
            self.cost.get_hessian_operator()
