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

    def test_prepare(self):
        problem = Problem(man=self.man, cost=self.cost)
        with self.assertRaises(ValueError):
            problem.prepare()

    def test_prepare_multiple(self):
        problem = Problem(man=self.man, cost=self.cost, arg=self.X)
        problem.prepare(need_grad=True)
        problem.prepare(need_grad=True, need_hess=True)
