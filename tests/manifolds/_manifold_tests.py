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

    def run_gradient_test(self):
        problem = pymanopt.Problem(self.manifold, self.cost)
        h, _, segment, poly = diagnostics.check_directional_derivative(problem)
        # Compute slope of linear regression line through points in linear
        # domain.
        x = np.log(h[segment])
        y = np.log(10) * np.polyval(poly, np.log10(np.e) * x)
        slope = scipy.stats.linregress(x, y).slope
        assert 1.995 <= slope <= 2.005
