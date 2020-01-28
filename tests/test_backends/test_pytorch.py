import unittest

import numpy as np
import numpy.random as rnd
import numpy.testing as np_testing
import torch

from pymanopt.function import PyTorch

from . import _backend_tests


class TestUnaryFunction(_backend_tests.TestUnaryFunction):
    def setUp(self):
        super().setUp()

        @PyTorch
        def cost(x):
            return torch.sum(x ** 2)

        self.cost = cost


class TestNaryFunction(_backend_tests.TestNaryFunction):
    def setUp(self):
        super().setUp()

        @PyTorch
        def cost(x, y):
            return torch.dot(x, y)

        self.cost = cost


class TestNaryParameterGrouping(_backend_tests.TestNaryParameterGrouping):
    def setUp(self):
        super().setUp()

        @PyTorch(("x", "y"), "z")
        def cost(x, y, z):
            return torch.sum(x ** 2 + y + z ** 3)

        self.cost = cost


class TestVector(_backend_tests.TestVector):
    def setUp(self):
        super().setUp()

        @PyTorch
        def cost(X):
            return torch.exp(torch.sum(X ** 2))

        self.cost = cost


class TestMatrix(unittest.TestCase):
    def setUp(self):
        np.seterr(all='raise')

        @PyTorch
        def cost(X):
            return torch.exp(torch.sum(X ** 2))
        self.cost = cost

        m = self.m = 10
        n = self.n = 15

        Y = self.Y = rnd.randn(m, n)
        A = self.A = rnd.randn(m, n)

        # Calculate correct cost and grad...
        self.correct_cost = np.exp(np.sum(Y ** 2))
        self.correct_grad = 2 * Y * np.exp(np.sum(Y ** 2))

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
        np_testing.assert_allclose(self.correct_cost, self.cost(self.Y))

    def test_grad(self):
        grad = self.cost.compute_gradient()
        np_testing.assert_allclose(self.correct_grad, grad(self.Y))

    def test_hessian(self):
        hess = self.cost.compute_hessian()

        # Now test hess
        np_testing.assert_allclose(self.correct_hess, hess(self.Y, self.A))


class TestTensor3(unittest.TestCase):
    def setUp(self):
        np.seterr(all='raise')

        @PyTorch
        def cost(X):
            return torch.exp(torch.sum(X ** 2))
        self.cost = cost

        n1 = self.n1 = 3
        n2 = self.n2 = 4
        n3 = self.n3 = 5

        Y = self.Y = rnd.randn(n1, n2, n3)
        A = self.A = rnd.randn(n1, n2, n3)

        # Calculate correct cost and grad...
        self.correct_cost = np.exp(np.sum(Y ** 2))
        self.correct_grad = 2 * Y * np.exp(np.sum(Y ** 2))

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

    def test_compile(self):
        np_testing.assert_allclose(self.correct_cost, self.cost(self.Y))

    def test_grad(self):
        grad = self.cost.compute_gradient()
        np_testing.assert_allclose(self.correct_grad, grad(self.Y))

    def test_hessian(self):
        hess = self.cost.compute_hessian()

        # Now test hess
        np_testing.assert_allclose(self.correct_hess, hess(self.Y, self.A))


class TestMixed(unittest.TestCase):
    # Test autograd on a tuple containing vector, matrix and tensor3.
    def setUp(self):
        np.seterr(all='raise')

        @PyTorch
        def f(x, y, z):
            return (torch.exp(torch.sum(x ** 2)) +
                    torch.exp(torch.sum(y ** 2)) +
                    torch.exp(torch.sum(z ** 2)))
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

        # Calculate correct grad
        g1 = 2 * y[0] * np.exp(np.sum(y[0] ** 2))
        g2 = 2 * y[1] * np.exp(np.sum(y[1] ** 2))
        g3 = 2 * y[2] * np.exp(np.sum(y[2] ** 2))

        self.correct_grad = (g1, g2, g3)

        # Calculate correct hess
        # 1. Vector
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

    def test_compile(self):
        np_testing.assert_allclose(self.correct_cost, self.cost(self.y))

    def test_grad(self):
        grad = self.cost.compute_gradient()
        g = grad(self.y)
        for k in range(len(g)):
            np_testing.assert_allclose(self.correct_grad[k], g[k])

    def test_hessian(self):
        hess = self.cost.compute_hessian()

        # Now test hess
        h = hess(self.y, self.a)
        for k in range(len(h)):
            np_testing.assert_allclose(self.correct_hess[k], h[k])
