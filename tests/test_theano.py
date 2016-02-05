import unittest

import numpy as np
import numpy.linalg as la
import numpy.random as rnd
import numpy.testing as np_testing

import theano.tensor as T

from pymanopt.tools.theano_functions import *


class TestMatrix(unittest.TestCase):
    def setUp(self):
        self.X = X = T.matrix()
        self.cost = T.exp(T.sum(X**2))

        m = self.m = 10
        n = self.n = 15

        Y = self.Y = rnd.randn(m, n)
        A = self.A = rnd.randn(m, n)

        # Calculate correct cost and grad...
        self.correct_cost = np.exp(np.sum(Y ** 2))
        self.correct_grad = correct_grad = ((2 * np.ones((m, n)) * Y *
                                             np.exp(np.sum(Y ** 2))))

        # ... and hess
        # First form hessian tensor H (4th order)
        Y1 = Y.reshape(m, n, 1, 1)
        Y2 = Y.reshape(1, 1, m, n)

        # Create an m x n x m x n array with diag[i,j,k,l] == 1 iff
        # (i == k and j == l), this is a 'diagonal' tensor.
        diag = np.eye(m * n).reshape(m, n, m, n)

        H = np.exp(np.sum(Y ** 2)) * (4 * Y1 * Y2 + 2 * diag)

        # Then 'right multiply' H by A
        Atensor = A.reshape(1, 1, m, n)

        self.correct_hess = np.sum(H * Atensor, axis=(2, 3))

    def test_compile(self):
        cost_compiled = compile(self.cost, self.X)
        np_testing.assert_allclose(self.correct_cost, cost_compiled(self.Y))

    def test_grad(self):
        grad = gradient(self.cost, self.X)
        np_testing.assert_allclose(self.correct_grad, grad(self.Y))

    def test_grad_hess(self):
        grad, hess = grad_hess(self.cost, self.X)

        # First test grad as above
        np_testing.assert_allclose(self.correct_grad, grad(self.Y))

        # Now test hess
        np_testing.assert_allclose(self.correct_hess, hess(self.Y, self.A))

    def test_grad_hess_no_Rop(self):
        # Break the Rop in T.exp
        def new_Rop(x, y):
            raise NotImplementedError

        T.exp.R_op = new_Rop

        # Rebuild graph to force recompile
        X = T.matrix()
        cost = T.exp(T.sum(X**2))

        # And check that all is still well
        grad, hess = grad_hess(cost, X)

        np_testing.assert_allclose(self.correct_hess, hess(self.Y, self.A))
