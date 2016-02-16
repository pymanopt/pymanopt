import unittest

import numpy as np
import numpy.linalg as la
import numpy.random as rnd
import numpy.testing as np_testing

import theano.tensor as T

import warnings

from pymanopt import Problem
from pymanopt.manifolds import Sphere


class TestProblem(unittest.TestCase):
    def setUp(self):
        self.X = X = T.vector()
        self.cost = T.exp(T.sum(X**2))

        n = self.n = 15

        self.man = Sphere(n)

    def test_compile(self):
        problem = Problem(man=self.man, cost=self.cost)
        with self.assertRaises(ValueError):
            problem.prepare()
