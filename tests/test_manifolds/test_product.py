import numpy as np
import numpy.testing as np_testing

from pymanopt.manifolds import Euclidean, Grassmann, Product, Sphere
from .._test import TestCase


class TestProductManifold(TestCase):
    def setUp(self):
        self.m = m = 100
        self.n = n = 50
        self.euclidean = Euclidean(m, n)
        self.sphere = Sphere(n)
        self.man = Product([self.euclidean, self.sphere])

    def test_dim(self):
        np_testing.assert_equal(self.man.dim, self.m*self.n+self.n-1)

    def test_typicaldist(self):
        np_testing.assert_equal(self.man.typicaldist,
                                np.sqrt((self.m*self.n)+np.pi**2))

    def test_dist(self):
        X = self.man.rand()
        Y = self.man.rand()
        np_testing.assert_equal(self.man.dist(X, Y),
                                np.sqrt(
                                    self.euclidean.dist(X[0], Y[0])**2 +
                                    self.sphere.dist(X[1], Y[1])**2))

    def test_tangent_vector_multiplication(self):
        # Regression test for https://github.com/pymanopt/pymanopt/issues/49.
        man = Product((Euclidean(12), Grassmann(12, 3)))
        x = man.rand()
        eta = man.randvec(x)
        np.float64(1.0) * eta

    # def test_inner(self):

    # def test_proj(self):

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
        Y = s.rand()
        Yexplog = s.exp(X, s.log(X, Y))
        np_testing.assert_almost_equal(s.dist(Y, Yexplog), 0)

    def test_log_exp_inverse(self):
        s = self.man
        X = s.rand()
        U = s.randvec(X)
        Ulogexp = s.log(X, s.exp(X, U))
        np_testing.assert_array_almost_equal(U[0], Ulogexp[0])
        np_testing.assert_array_almost_equal(U[1], Ulogexp[1])

    def test_pairmean(self):
        s = self.man
        X = s.rand()
        Y = s.rand()
        Z = s.pairmean(X, Y)
        np_testing.assert_array_almost_equal(s.dist(X, Z), s.dist(Y, Z))
