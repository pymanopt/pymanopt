import autograd.numpy as np

import pymanopt
from pymanopt.tools import diagnostics

from .._test import TestCase


class ManifoldTestCase(TestCase):
    def setUp(self):
        random_point = self.manifold.random_point()

        @pymanopt.function.autograd(self.manifold)
        def cost(point):
            return np.linalg.norm(point - random_point) ** 2

        self.cost = cost

    def run_gradient_approximation_test(self):
        problem = pymanopt.Problem(self.manifold, self.cost)
        *_, (slope, *_) = diagnostics.check_directional_derivative(problem)
        assert 1.95 <= slope <= 2.05

    def run_hessian_approximation_test(self):
        problem = pymanopt.Problem(self.manifold, self.cost)
        _, error, _, (slope, *_) = diagnostics.check_directional_derivative(
            problem, use_quadratic_model=True
        )
        assert np.allclose(np.linalg.norm(error), 0) or (2.95 <= slope <= 3.05)
