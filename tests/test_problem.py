import unittest

import numpy as np
from numpy import random as rnd
import numpy.testing as np_testing
import theano.tensor as T

from pymanopt import Problem, TheanoFunction
from pymanopt.manifolds import Sphere


class TestProblem(unittest.TestCase):
    def setUp(self):
        self.X = X = T.vector()
        self.cost = TheanoFunction(T.exp(T.sum(X**2)), X)

        n = self.n = 15

        self.man = Sphere(n)

    def test_prepare(self):
        problem = Problem(self.man, self.cost)
        x = rnd.randn(self.n)
        np_testing.assert_allclose(2 * x * np.exp(np.sum(x ** 2)),
                                   problem.egrad(x))
