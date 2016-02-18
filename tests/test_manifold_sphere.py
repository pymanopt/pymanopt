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
        self.man = Sphere(m, n)

    # def test_dim(self):

    # def test_typicaldist(self):

    # def test_dist(self):

    # def test_inner(self):

    def test_proj(self):
        #  Construct a random point X on the manifold.
        X = rnd.randn(self.m, self.n)
        X /= la.norm(X, "fro")

        #  Construct a vector H in the ambient space.
        H = rnd.randn(self.m, self.n)

        #  Compare the projections.
        np_testing.assert_array_almost_equal(H - X * np.trace(X.T.dot(H)),
                                             self.man.proj(X, H))

    # def test_ehess2rhess(self):

    # def test_retr(self):

    # def test_egrad2rgrad(self):

    # def test_norm(self):

    # def test_rand(self):

    # def test_randvec(self):

    # def test_transp(self):

    def test_exp_log_inverse(self):
        s = self.man
        X = s.rand()
        U = s.randvec(X)
        Uexplog = s.exp(X, s.log(X, U))
        np_testing.assert_array_almost_equal(U, Uexplog)

    def test_log_exp_inverse(self):
        s = self.man
        X = s.rand()
        U = s.randvec(X)
        Ulogexp = s.log(X, s.exp(X, U))
        np_testing.assert_array_almost_equal(U, Ulogexp)

    def test_pairmean(self):
        s = self.man
        X = s.rand()
        Y = s.rand()
        Z = s.pairmean(X, Y)
        np_testing.assert_array_almost_equal(s.dist(X, Z), s.dist(Y, Z))
