import autograd.numpy as np
from nose2.tools import params

import pymanopt
import pymanopt.optimizers
from pymanopt.manifolds import Sphere

from .._test import TestCase


class TestSolvers(TestCase):
    def setUp(self):
        n = 32
        matrix = np.random.normal(size=(n, n))
        self.manifold = manifold = Sphere(n)

        @pymanopt.function.autograd(manifold)
        def cost(point):
            return -point.T @ matrix @ point

        self.problem = pymanopt.Problem(manifold, cost)

    @params(*pymanopt.optimizers.__all__)
    def test_closest_unit_norm_column_approximation(self, optimizer_name):
        optimizer = getattr(pymanopt.optimizers, optimizer_name)(
            max_iterations=50, verbosity=0
        )
        optimizer.run(self.problem)
