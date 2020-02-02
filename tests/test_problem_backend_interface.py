import autograd.numpy as np
import numpy.testing as np_testing

import pymanopt
from pymanopt.manifolds import Euclidean, FixedRankEmbedded, Product
from ._test import TestCase


class TestProblemBackendInterface(TestCase):
    def setUp(self):
        self.m = m = 20
        self.n = n = 10
        self.rank = rank = 3

        A = np.random.randn(m, n)

        @pymanopt.function.Autograd
        def cost(u, s, vt, x):
            return np.linalg.norm(((u * s) @ vt - A) @ x) ** 2

        self.cost = cost
        self.gradient = self.cost.compute_gradient()
        self.hvp = self.cost.compute_hessian_vector_product()

        self.manifold = Product([FixedRankEmbedded(m, n, rank), Euclidean(n)])
        self.problem = pymanopt.Problem(self.manifold, self.cost)

    def test_cost_function(self):
        (u, s, vt), x = self.manifold.rand()
        self.cost(u, s, vt, x)

    def test_gradient(self):
        (u, s, vt), x = self.manifold.rand()
        gu, gs, gvt, gx = self.gradient(u, s, vt, x)
        self.assertEqual(gu.shape, (self.m, self.rank))
        self.assertEqual(gs.shape, (self.rank,))
        self.assertEqual(gvt.shape, (self.rank, self.n))
        self.assertEqual(gx.shape, (self.n,))

    def test_hessian_vector_product(self):
        (u, s, vt), x = self.manifold.rand()
        (a, b, c), d = self.manifold.rand()
        hu, hs, hvt, hx = self.hvp(u, s, vt, x, a, b, c, d)
        self.assertEqual(hu.shape, (self.m, self.rank))
        self.assertEqual(hs.shape, (self.rank,))
        self.assertEqual(hvt.shape, (self.rank, self.n))
        self.assertEqual(hx.shape, (self.n,))

    def test_problem_cost(self):
        cost = self.problem.cost
        X = self.manifold.rand()
        (u, s, vt), x = X
        np_testing.assert_allclose(cost(X), self.cost(u, s, vt, x))

    def test_problem_egrad(self):
        egrad = self.problem.egrad
        X = self.manifold.rand()
        (u, s, vt), x = X
        G = egrad(X)
        (gu, gs, gvt), gx = G
        for ga, gb in zip((gu, gs, gvt, gx), self.gradient(u, s, vt, x)):
            np_testing.assert_allclose(ga, gb)

    def test_problem_hessian_vector_product(self):
        ehess = self.problem.ehess
        X = self.manifold.rand()
        U = self.manifold.rand()
        H = ehess(X, U)

        (u, s, vt), x = X
        (a, b, c), d = U

        (hu, hs, hvt), hx = H
        for ha, hb in zip((hu, hs, hvt, hx),
                          self.hvp(u, s, vt, x, a, b, c, d)):
            np_testing.assert_allclose(ha, hb)
