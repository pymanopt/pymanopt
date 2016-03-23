import unittest

import numpy.linalg as la
import numpy.random as rnd
import numpy.testing as np_testing

import warnings

import autograd.numpy as np
from pymanopt.tools.autodiff import AutogradBackend


class TestVector(unittest.TestCase):
    def setUp(self):
        np.seterr(all='raise')
        self.X = None
        self.cost = lambda X: np.exp(np.sum(X**2))

        n = self.n = 15

        Y = self.Y = rnd.randn(n)
        A = self.A = rnd.randn(n)

        # Calculate correct cost and grad...
        self.correct_cost = np.exp(np.sum(Y ** 2))
        self.correct_grad = correct_grad = 2 * Y * np.exp(np.sum(Y ** 2))

        # ... and hess
        # First form hessian matrix H
        # Convert Y and A into matrices (row vectors)
        Ymat = np.matrix(Y)
        Amat = np.matrix(A)

        diag = np.eye(n)

        H = np.exp(np.sum(Y ** 2)) * (4 * Ymat.T.dot(Ymat) + 2 * diag)

        # Then 'left multiply' H by A
        self.correct_hess = np.squeeze(np.array(Amat.dot(H)))

        self.backend = AutogradBackend()

    def test_compile(self):
        cost_compiled = self.backend.compile_function(self.cost, self.X)
        np_testing.assert_allclose(self.correct_cost, cost_compiled(self.Y))

    def test_grad(self):
        grad = self.backend.compute_gradient(self.cost, self.X)
        np_testing.assert_allclose(self.correct_grad, grad(self.Y))

    def test_hessian(self):
        hess = self.backend.compute_hessian(self.cost, self.X)

        # Now test hess
        np_testing.assert_allclose(self.correct_hess, hess(self.Y, self.A))


class TestMatrix(unittest.TestCase):
    def setUp(self):
        np.seterr(all='raise')

        self.X = None
        self.cost = lambda X: np.exp(np.sum(X**2))

        m = self.m = 10
        n = self.n = 15

        Y = self.Y = rnd.randn(m, n)
        A = self.A = rnd.randn(m, n)

        # Calculate correct cost and grad...
        self.correct_cost = np.exp(np.sum(Y ** 2))
        self.correct_grad = correct_grad = 2 * Y * np.exp(np.sum(Y ** 2))

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

        self.backend = AutogradBackend()

    def test_compile(self):
        cost_compiled = self.backend.compile_function(self.cost, self.X)
        np_testing.assert_allclose(self.correct_cost, cost_compiled(self.Y))

    def test_grad(self):
        grad = self.backend.compute_gradient(self.cost, self.X)
        np_testing.assert_allclose(self.correct_grad, grad(self.Y))

    def test_hessian(self):
        hess = self.backend.compute_hessian(self.cost, self.X)

        # Now test hess
        np_testing.assert_allclose(self.correct_hess, hess(self.Y, self.A))


class TestTensor3(unittest.TestCase):
    def setUp(self):
        np.seterr(all='raise')

        self.X = None
        self.cost = lambda X: np.exp(np.sum(X**2))

        n1 = self.n1 = 3
        n2 = self.n2 = 4
        n3 = self.n3 = 5

        Y = self.Y = rnd.randn(n1, n2, n3)
        A = self.A = rnd.randn(n1, n2, n3)

        # Calculate correct cost and grad...
        self.correct_cost = np.exp(np.sum(Y ** 2))
        self.correct_grad = correct_grad = 2 * Y * np.exp(np.sum(Y ** 2))

        # ... and hess
        # First form hessian tensor H (6th order)
        Y1 = Y.reshape(n1, n2, n3, 1, 1, 1)
        Y2 = Y.reshape(1, 1, 1, n1, n2, n3)

        # Create an n1 x n2 x n3 x n1 x n2 x n3 diagonal tensor
        diag = np.eye(n1 * n2 * n3).reshape(n1, n2, n3, n1, n2, n3)

        H = np.exp(np.sum(Y ** 2)) * (4 * Y1 * Y2 + 2 * diag)

        # Then 'right multiply' H by A
        Atensor = A.reshape(1, 1, 1, n1, n2, n3)

        self.correct_hess = np.sum(H * Atensor, axis=(3, 4, 5))

        self.backend = AutogradBackend()

    def test_compile(self):
        cost_compiled = self.backend.compile_function(self.cost, self.X)
        np_testing.assert_allclose(self.correct_cost, cost_compiled(self.Y))

    def test_grad(self):
        grad = self.backend.compute_gradient(self.cost, self.X)
        np_testing.assert_allclose(self.correct_grad, grad(self.Y))

    def test_hessian(self):
        hess = self.backend.compute_hessian(self.cost, self.X)

        # Now test hess
        np_testing.assert_allclose(self.correct_hess, hess(self.Y, self.A))


class TestMixed(unittest.TestCase):
    # Test autograd on a tuple containing vector, matrix and tensor3.
    def setUp(self):
        np.seterr(all='raise')

        def f(x):
            return (np.exp(np.sum(x[0]**2)) + np.exp(np.sum(x[1]**2)) +
                    np.exp(np.sum(x[2]**2)))

        self.cost = f

        n1 = self.n1 = 3
        n2 = self.n2 = 4
        n3 = self.n3 = 5
        n4 = self.n4 = 6
        n5 = self.n5 = 7
        n6 = self.n6 = 8

        self.y = y = (rnd.randn(n1), rnd.randn(n2, n3), rnd.randn(n4, n5, n6))
        self.a = a = (rnd.randn(n1), rnd.randn(n2, n3), rnd.randn(n4, n5, n6))

        self.correct_cost = f(y)

        # CALCULATE CORRECT GRAD
        g1 = 2 * y[0] * np.exp(np.sum(y[0] ** 2))
        g2 = 2 * y[1] * np.exp(np.sum(y[1] ** 2))
        g3 = 2 * y[2] * np.exp(np.sum(y[2] ** 2))

        self.correct_grad = (g1, g2, g3)

        # CALCULATE CORRECT HESS
        # 1. VECTOR
        Ymat = np.matrix(y[0])
        Amat = np.matrix(a[0])

        diag = np.eye(n1)

        H = np.exp(np.sum(y[0] ** 2)) * (4 * Ymat.T.dot(Ymat) + 2 * diag)

        # Then 'left multiply' H by A
        h1 = np.array(Amat.dot(H)).flatten()

        # 2. MATRIX
        # First form hessian tensor H (4th order)
        Y1 = y[1].reshape(n2, n3, 1, 1)
        Y2 = y[1].reshape(1, 1, n2, n3)

        # Create an m x n x m x n array with diag[i,j,k,l] == 1 iff
        # (i == k and j == l), this is a 'diagonal' tensor.
        diag = np.eye(n2 * n3).reshape(n2, n3, n2, n3)

        H = np.exp(np.sum(y[1] ** 2)) * (4 * Y1 * Y2 + 2 * diag)

        # Then 'right multiply' H by A
        Atensor = a[1].reshape(1, 1, n2, n3)

        h2 = np.sum(H * Atensor, axis=(2, 3))

        # 3. Tensor3
        # First form hessian tensor H (6th order)
        Y1 = y[2].reshape(n4, n5, n6, 1, 1, 1)
        Y2 = y[2].reshape(1, 1, 1, n4, n5, n6)

        # Create an n1 x n2 x n3 x n1 x n2 x n3 diagonal tensor
        diag = np.eye(n4 * n5 * n6).reshape(n4, n5, n6, n4, n5, n6)

        H = np.exp(np.sum(y[2] ** 2)) * (4 * Y1 * Y2 + 2 * diag)

        # Then 'right multiply' H by A
        Atensor = a[2].reshape(1, 1, 1, n4, n5, n6)

        h3 = np.sum(H * Atensor, axis=(3, 4, 5))

        self.correct_hess = (h1, h2, h3)
        self.backend = AutogradBackend()

    def test_compile(self):
        cost_compiled = self.backend.compile_function(self.cost, None)
        np_testing.assert_allclose(self.correct_cost, cost_compiled(self.y))

    def test_grad(self):
        grad = self.backend.compute_gradient(self.cost, None)
        for k in range(len(grad(self.y))):
            np_testing.assert_allclose(self.correct_grad[k], grad(self.y)[k])

    def test_hessian(self):
        hess = self.backend.compute_hessian(self.cost, None)

        # Now test hess
        for k in range(len(hess(self.y, self.a))):
            np_testing.assert_allclose(self.correct_hess[k], hess(self.y,
                                                                  self.a)[k])
