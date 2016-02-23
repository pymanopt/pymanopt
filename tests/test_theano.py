import unittest

import numpy as np
import numpy.linalg as la
import numpy.random as rnd
import numpy.testing as np_testing

import theano.tensor as T

import warnings

from pymanopt.tools.autodiff import TheanoBackend


class TestVector(unittest.TestCase):
    def setUp(self):
        self.X = X = T.vector()
        self.cost = T.exp(T.sum(X**2))

        n = self.n = 15

        Y = self.Y = rnd.randn(n)
        A = self.A = rnd.randn(n)

        # Calculate correct cost and grad...
        self.correct_cost = np.exp(np.sum(Y ** 2))
        self.correct_grad = correct_grad = 2 * Y * np.exp(np.sum(Y ** 2))

        # ... and hess
        # First form hessian matrix H
        # Convert Y and A into matrices (column vectors)
        Ymat = np.matrix(Y)
        Amat = np.matrix(A)

        diag = np.eye(n)

        H = np.exp(np.sum(Y ** 2)) * (4 * Ymat.T.dot(Ymat) + 2 * diag)

        # Then 'right multiply' H by A
        self.correct_hess = np.array(Amat.dot(H)).squeeze()

        self.backend = TheanoBackend()

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

    def test_hessian_no_Rop(self):
        # Break the Rop in T.exp
        def new_Rop(x, y):
            raise NotImplementedError

        T.exp.R_op = new_Rop

        # Rebuild graph to force recompile
        X = T.vector()
        cost = T.exp(T.sum(X**2))

        # And check that all is still well
        hess = self.backend.compute_hessian(cost, X)

        np_testing.assert_allclose(self.correct_hess, hess(self.Y, self.A))


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

        self.backend = TheanoBackend()

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

    def test_hessian_no_Rop(self):
        # Break the Rop in T.exp
        Rop = T.exp.R_op

        def broken_Rop(x, y):
            raise NotImplementedError
        T.exp.R_op = broken_Rop

        # Rebuild graph to force recompile
        X = T.matrix()
        cost = T.exp(T.sum(X**2))

        # And check that all is still well
        hess = self.backend.compute_hessian(cost, X)

        np_testing.assert_allclose(self.correct_hess, hess(self.Y, self.A))

        # Fix broken Rop
        T.exp.R_op = Rop

    def test_hessian_nodependence(self):
        X = T.matrix()
        cost = T.sum(X)

        with warnings.catch_warnings(record=True) as w:
            # The following should emit a warning
            hess = self.backend.compute_hessian(cost, X)

            assert len(w) == 1
            assert "not part of the computational graph" in str(w[-1].message)


class TestTensor3(unittest.TestCase):
    def setUp(self):
        self.X = X = T.tensor3()
        self.cost = T.exp(T.sum(X**2))

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

        self.backend = TheanoBackend()

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

    def test_hessian_no_Rop(self):
        # Break the Rop in T.exp
        def new_Rop(x, y):
            raise NotImplementedError

        T.exp.R_op = new_Rop

        # Rebuild graph to force recompile
        X = T.tensor3()
        cost = T.exp(T.sum(X**2))

        # And check that all is still well
        hess = self.backend.compute_hessian(cost, X)

        np_testing.assert_allclose(self.correct_hess, hess(self.Y, self.A))


class TestMixed(unittest.TestCase):
    # Test autograd on a tuple containing vector, matrix and tensor3.
    def setUp(self):
        x = T.vector()
        y = T.matrix()
        z = T.tensor3()
        f = T.exp(T.sum(x**2)) + T.exp(T.sum(y**2)) + T.exp(T.sum(z**2))

        self.cost = f
        self.arg = [x, y, z]

        n1 = self.n1 = 3
        n2 = self.n2 = 4
        n3 = self.n3 = 5
        n4 = self.n4 = 6
        n5 = self.n5 = 7
        n6 = self.n6 = 8

        self.y = y = (rnd.randn(n1), rnd.randn(n2, n3), rnd.randn(n4, n5, n6))
        self.a = a = (rnd.randn(n1), rnd.randn(n2, n3), rnd.randn(n4, n5, n6))

        self.correct_cost = (np.exp(np.sum(y[0]**2)) +
                             np.exp(np.sum(y[1]**2)) +
                             np.exp(np.sum(y[2]**2)))

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
        self.backend = TheanoBackend()

    def test_compile(self):
        cost_compiled = self.backend.compile_function(self.cost, self.arg)
        np_testing.assert_allclose(self.correct_cost, cost_compiled(self.y))

    def test_grad(self):
        grad = self.backend.compute_gradient(self.cost, self.arg)
        for k in range(len(grad(self.y))):
            np_testing.assert_allclose(self.correct_grad[k], grad(self.y)[k])

    def test_hessian(self):
        hess = self.backend.compute_hessian(self.cost, self.arg)

        # Now test hess
        for k in range(len(hess(self.y, self.a))):
            np_testing.assert_allclose(self.correct_hess[k], hess(self.y,
                                                                  self.a)[k])
