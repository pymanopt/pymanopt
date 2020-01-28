from pymanopt.manifolds import PSDFixedRankComplex
from .._test import TestCase


class TestPSDFixedRankComplexManifold(TestCase):
    def test_constructor(self):
        n = 50
        k = 10
        PSDFixedRankComplex(n, k)

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

    # def test_exp_log_inverse(self):
        # s = self.man
        # X = s.rand()
        # U = s.randvec(X)
        # Uexplog = s.exp(X, s.log(X, U))
        # np_testing.assert_array_almost_equal(U, Uexplog)

    # def test_log_exp_inverse(self):
        # s = self.man
        # X = s.rand()
        # U = s.randvec(X)
        # Ulogexp = s.log(X, s.exp(X, U))
        # np_testing.assert_array_almost_equal(U, Ulogexp)

    # def test_pairmean(self):
        # s = self.man
        # X = s.rand()
        # Y = s.rand()
        # Z = s.pairmean(X, Y)
        # np_testing.assert_array_almost_equal(s.dist(X, Z), s.dist(Y, Z))
