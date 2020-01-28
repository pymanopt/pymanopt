import unittest

import numpy as np
import theano.tensor as T
from numpy import random as rnd, testing as np_testing

from pymanopt.function import Theano

from . import _backend_tests


class TestUnaryFunction(_backend_tests.TestUnaryFunction):
    def setUp(self):
        super().setUp()

        x = T.vector()

        @Theano(x)
        def cost(x):
            return T.sum(x ** 2)

        self.cost = cost


class TestNaryFunction(_backend_tests.TestNaryFunction):
    def setUp(self):
        super().setUp()

        x = T.vector()
        y = T.vector()

        @Theano(x, y)
        def cost(x, y):
            return T.dot(x, y)

        self.cost = cost


class TestNaryParameterGrouping(_backend_tests.TestNaryParameterGrouping):
    def setUp(self):
        super().setUp()

        x = T.vector()
        y = T.vector()
        z = T.vector()

        @Theano((x, y), z)
        def cost(x, y, z):
            return T.sum(x ** 2 + y + z ** 3)

        self.cost = cost


class TestVector(unittest.TestCase):
    def setUp(self):
        X = T.vector()

        @Theano(X)
        def cost(X):
            return T.exp(T.sum(X ** 2))
        self.cost = cost

        n = self.n = 15

        Y = self.Y = rnd.randn(n)
        A = self.A = rnd.randn(n)

        # Calculate correct cost and grad...
        self.correct_cost = np.exp(np.sum(Y ** 2))
        self.correct_grad = 2 * Y * np.exp(np.sum(Y ** 2))

        # ... and hess
        # First form hessian matrix H
        # Convert Y and A into matrices (column vectors)
        Ymat = np.matrix(Y)
        Amat = np.matrix(A)

        diag = np.eye(n)

        H = np.exp(np.sum(Y ** 2)) * (4 * Ymat.T.dot(Ymat) + 2 * diag)

        # Then 'right multiply' H by A
        self.correct_hess = np.array(Amat.dot(H)).squeeze()

    def test_compile(self):
        np_testing.assert_allclose(self.correct_cost, self.cost(self.Y))

    def test_grad(self):
        grad = self.cost.compute_gradient()
        np_testing.assert_allclose(self.correct_grad, grad(self.Y))

    def test_hessian(self):
        hess = self.cost.compute_hessian()

        # Now test hess
        np_testing.assert_allclose(self.correct_hess, hess(self.Y, self.A))

    def test_hessian_no_Rop(self):
        # Break the Rop in T.exp
        Rop = T.exp.R_op

        def new_Rop(x, y):
            raise NotImplementedError
        T.exp.R_op = new_Rop

        # Rebuild graph to force recompile
        X = T.vector()

        @Theano(X)
        def cost(X):
            return T.exp(T.sum(X ** 2))

        # And check that all is still well
        hess = cost.compute_hessian()

        np_testing.assert_allclose(self.correct_hess, hess(self.Y, self.A))

        # Fix broken Rop
        T.exp.R_op = Rop


class TestMatrix(unittest.TestCase):
    def setUp(self):
        X = T.matrix()

        @Theano(X)
        def cost(X):
            return T.exp(T.sum(X ** 2))
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

    def test_hessian_no_Rop(self):
        # Break the Rop in T.exp
        Rop = T.exp.R_op

        def broken_Rop(x, y):
            raise NotImplementedError
        T.exp.R_op = broken_Rop

        # Rebuild graph to force recompile
        X = T.matrix()

        @Theano(X)
        def cost(X):
            return T.exp(T.sum(X ** 2))

        # And check that all is still well
        hess = cost.compute_hessian()

        np_testing.assert_allclose(self.correct_hess, hess(self.Y, self.A))

        # Fix broken Rop
        T.exp.R_op = Rop


class TestTensor3(unittest.TestCase):
    def setUp(self):
        X = T.tensor3()

        @Theano(X)
        def cost(X):
            return T.exp(T.sum(X ** 2))
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

    def test_hessian_no_Rop(self):
        # Break the Rop in T.exp
        Rop = T.exp.R_op

        def new_Rop(x, y):
            raise NotImplementedError
        T.exp.R_op = new_Rop

        # Rebuild graph to force recompile
        X = T.tensor3()

        @Theano(X)
        def cost(X):
            return T.exp(T.sum(X ** 2))

        # And check that all is still well
        hess = cost.compute_hessian()

        np_testing.assert_allclose(self.correct_hess, hess(self.Y, self.A))

        # Fix broken Rop
        T.exp.R_op = Rop


class TestMixed(unittest.TestCase):
    # Test autograd on a tuple containing vector, matrix and tensor3, i.e.,
    # test a cost function defined on a product manifold.
    def setUp(self):
        x = T.vector()
        y = T.matrix()
        z = T.tensor3()
        f = T.exp(T.sum(x ** 2)) + T.exp(T.sum(y ** 2)) + T.exp(T.sum(z ** 2))

        @Theano(x, y, z)
        def cost(x, y, z):
            return f
        self.cost = cost

        n1 = self.n1 = 3
        n2 = self.n2 = 4
        n3 = self.n3 = 5
        n4 = self.n4 = 6
        n5 = self.n5 = 7
        n6 = self.n6 = 8

        self.y = y = (rnd.randn(n1), rnd.randn(n2, n3), rnd.randn(n4, n5, n6))
        self.a = a = (rnd.randn(n1), rnd.randn(n2, n3), rnd.randn(n4, n5, n6))

        self.correct_cost = (np.exp(np.sum(y[0] ** 2)) +
                             np.exp(np.sum(y[1] ** 2)) +
                             np.exp(np.sum(y[2] ** 2)))

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

    def test_compile(self):
        np_testing.assert_allclose(self.correct_cost, self.cost(self.y))

    def test_grad(self):
        grad = self.cost.compute_gradient()
        g = grad(self.y)
        for k in range(len(g)):
            np_testing.assert_allclose(self.correct_grad[k], g[k])

    def test_hessian(self):
        hess = self.cost.compute_hessian()
        h = hess(self.y, self.a)
        for k in range(len(h)):
            np_testing.assert_allclose(self.correct_hess[k], h[k])

    def test_hessian_no_Rop(self):
        # Break the Rop in T.exp
        Rop = T.exp.R_op

        def new_Rop(x, y):
            raise NotImplementedError
        T.exp.R_op = new_Rop

        # Rebuild graph to force recompile
        x = T.vector()
        y = T.matrix()
        z = T.tensor3()
        f = T.exp(T.sum(x ** 2)) + T.exp(T.sum(y ** 2)) + T.exp(T.sum(z ** 2))

        # Alternative use of `Theano' in decorator notation.
        cost = Theano(x, y, z)(lambda x, y, z: f)

        # And check that all is still well
        hess = cost.compute_hessian()

        h = hess(self.y, self.a)
        for k in range(len(h)):
            np_testing.assert_allclose(self.correct_hess[k], h[k])

        # Fix broken Rop
        T.exp.R_op = Rop
