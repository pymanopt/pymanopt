import numpy as np
import tensorflow as tf
from numpy import random as rnd
from numpy import testing as np_testing

import pymanopt
from pymanopt.manifolds import Product, Sphere, Stiefel
from pymanopt.solvers import TrustRegions

from ._test import TestCase


class TestProblem(TestCase):
    def setUp(self):
        self.n = 15
        self.man = Sphere(self.n)

        @pymanopt.function.TensorFlow(self.man)
        def cost(X):
            return tf.exp(tf.reduce_sum(X ** 2))

        self.cost = cost

    def test_prepare(self):
        problem = pymanopt.Problem(self.man, self.cost)
        x = rnd.randn(self.n)
        np_testing.assert_allclose(
            2 * x * np.exp(np.sum(x ** 2)), problem.egrad(x)
        )

    def test_attribute_override(self):
        problem = pymanopt.Problem(self.man, self.cost)
        with self.assertRaises(ValueError):
            problem.verbosity = "0"
        with self.assertRaises(ValueError):
            problem.verbosity = -1
        problem.verbosity = 2
        with self.assertRaises(AttributeError):
            problem.manifold = None

    def test_vararg_cost_on_product(self):
        shape = (3, 3)
        manifold = Product([Stiefel(*shape)] * 2)

        @pymanopt.function.TensorFlow(manifold)
        def cost(*args):
            X, Y = args
            return tf.reduce_sum(X) + tf.reduce_sum(Y)

        problem = pymanopt.Problem(manifold=manifold, cost=cost)
        solver = TrustRegions(maxiter=1)
        Xopt, Yopt = solver.solve(problem)
        self.assertEqual(Xopt.shape, (3, 3))
        self.assertEqual(Yopt.shape, (3, 3))
