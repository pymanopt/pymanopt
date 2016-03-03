import unittest

import numpy as np
import numpy.linalg as la
import numpy.random as rnd
import numpy.testing as np_testing

from pymanopt.manifolds import PositiveDefinite
from pymanopt.tools.multi import *

class TestPositiveDefiniteManifold(unittest.TestCase):
    def setUp(self):
        self.n = n = 10
        self.k = k = 3
        self.man = PositiveDefinite(n, k)

    def test_dim(self):
        man = self.man
        n = self.n
        k = self.k
        np_testing.assert_equal(man.dim, 0.5 * k * n * (n+1))

class TestPositiveDefiniteManifoldSingle(unittest.TestCase):
    def setUp(self):
        self.n = n = 15
        self.man = PositiveDefinite(n)
