import numpy as np
import theano.tensor as T
from numpy import random as rnd, testing as np_testing

import pymanopt
from pymanopt.manifolds import Sphere
from ._test import TestCase


class TestProblem(TestCase):
    def setUp(self):
        X = T.vector()

        @pymanopt.function.Theano(X)
        def cost(X):
            return T.exp(T.sum(X ** 2))

        self.cost = cost

        n = self.n = 15

        self.man = Sphere(n)

    def test_prepare(self):
        problem = pymanopt.Problem(self.man, self.cost)
        x = rnd.randn(self.n)
        np_testing.assert_allclose(2 * x * np.exp(np.sum(x ** 2)),
                                   problem.egrad(x))

    def test_attribute_override(self):
        problem = pymanopt.Problem(self.man, self.cost)
        with self.assertRaises(ValueError):
            problem.verbosity = "0"
        with self.assertRaises(ValueError):
            problem.verbosity = -1
        problem.verbosity = 2
        with self.assertRaises(AttributeError):
            problem.manifold = None
