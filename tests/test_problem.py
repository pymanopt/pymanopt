import numpy as np
import tensorflow as tf
from numpy import random as rnd
from numpy import testing as np_testing

import pymanopt
from pymanopt.manifolds import Product, Sphere, Stiefel
from pymanopt.optimizers import TrustRegions

from ._test import TestCase


class TestProblem(TestCase):
    def setUp(self):
        self.n = 15
        self.man = Sphere(self.n)

        @pymanopt.function.tensorflow(self.man)
        def cost(X):
            return tf.exp(tf.reduce_sum(X**2))

        self.cost = cost

    def test_prepare(self):
        problem = pymanopt.Problem(self.man, self.cost)
        x = rnd.randn(self.n)
        np_testing.assert_allclose(
            2 * x * np.exp(np.sum(x**2)), problem.egrad(x)
        )

    def test_attribute_override(self):
        problem = pymanopt.Problem(self.man, self.cost)
        with self.assertRaises(AttributeError):
            problem.manifold = None

    def test_vararg_cost_on_product(self):
        shape = (3, 3)
        manifold = Product([Stiefel(*shape)] * 2)

        @pymanopt.function.tensorflow(manifold)
        def cost(*args):
            X, Y = args
            return tf.reduce_sum(X) + tf.reduce_sum(Y)

        problem = pymanopt.Problem(manifold=manifold, cost=cost)
        optimizer = TrustRegions(max_iterations=1)
        Xopt, Yopt = optimizer.run(problem)
        self.assertEqual(Xopt.shape, (3, 3))
        self.assertEqual(Yopt.shape, (3, 3))
