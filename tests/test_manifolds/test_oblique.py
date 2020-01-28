import numpy.testing as np_testing

from pymanopt.manifolds import Oblique
from .._test import TestCase


class TestObliqueManifold(TestCase):
    def setUp(self):
        self.m = m = 100
        self.n = n = 50
        self.man = Oblique(m, n)

    # def test_dim(self):

    # def test_typicaldist(self):

    # def test_dist(self):

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
        x = s.rand()
        y = s.rand()
        u = s.log(x, y)
        z = s.exp(x, u)
        np_testing.assert_almost_equal(0, s.dist(y, z), decimal=6)

    def test_log_exp_inverse(self):
        s = self.man
        x = s.rand()
        u = s.randvec(x)
        y = s.exp(x, u)
        v = s.log(x, y)
        # Check that the manifold difference between the tangent vectors u and
        # v is 0
        np_testing.assert_almost_equal(0, s.norm(x, u - v))

    def test_pairmean(self):
        s = self.man
        X = s.rand()
        Y = s.rand()
        Z = s.pairmean(X, Y)
        np_testing.assert_array_almost_equal(s.dist(X, Z), s.dist(Y, Z))
