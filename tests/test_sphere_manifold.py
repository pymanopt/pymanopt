import unittest

import numpy as np
import numpy.linalg as la
import numpy.random as rnd
import numpy.testing as np_testing

from pymanopt.manifolds import Sphere


class TestSphereManifold(unittest.TestCase):
    def setUp(self):
        self.m = m = 100
        self.n = n = 50
        self.sphere = Sphere(m, n)

    def test_proj(self):
        # Construct a random point (X) on the manifold.
        X = rnd.randn(self.m, self.n)
        X /= la.norm(X, "fro")

        # Construct a vector H) in the ambient space.
        H = rnd.randn(self.m, self.n)

        # Compare the projections.
        np_testing.assert_array_almost_equal(H - X * np.trace(X.T.dot(H)),
                                             self.sphere.proj(X, H))
