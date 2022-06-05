import autograd.numpy as np
import scipy.stats

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
        # Approximate the cost function with a 1st order model.
        h, _, segment, poly = diagnostics.check_directional_derivative(problem)
        x = np.log(h[segment])
        y = np.log(10) * np.polyval(poly, np.log10(np.e) * x)
        slope = scipy.stats.linregress(x, y).slope
        assert 1.95 <= slope <= 2.05

    def run_hessian_approximation_test(self):
        problem = pymanopt.Problem(self.manifold, self.cost)
        # Approximate the cost function with a 2nd order model.
        h, error, segment, poly = diagnostics.check_directional_derivative(
            problem, use_quadratic_model=True
        )
        x = np.log(h[segment])
        y = np.log(10) * np.polyval(poly, np.log10(np.e) * x)
        slope = scipy.stats.linregress(x, y).slope
        assert np.allclose(np.linalg.norm(error), 0) or (2.95 <= slope <= 3.05)
