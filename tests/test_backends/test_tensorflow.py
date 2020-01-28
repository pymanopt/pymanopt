import os
import unittest

import numpy as np
import numpy.random as rnd
import numpy.testing as np_testing
import tensorflow as tf

from pymanopt.function import TensorFlow

from . import _backend_tests

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class TestUnaryFunction(_backend_tests.TestUnaryFunction):
    def setUp(self):
        super().setUp()

        x = tf.Variable(tf.zeros(self.n, dtype=np.float64), name="x")

        @TensorFlow(x)
        def cost(x):
            return tf.reduce_sum(x ** 2)

        self.cost = cost


class TestNaryFunction(_backend_tests.TestNaryFunction):
    def setUp(self):
        super().setUp()

        n = self.n

        x = tf.Variable(tf.zeros(n, dtype=np.float64), name="x")
        y = tf.Variable(tf.zeros(n, dtype=np.float64), name="y")

        @TensorFlow(x, y)
        def cost(x, y):
            return tf.tensordot(x, y, axes=1)

        self.cost = cost


class TestNaryParameterGrouping(_backend_tests.TestNaryParameterGrouping):
    def setUp(self):
        super().setUp()

        n = self.n

        x = tf.Variable(tf.zeros(n, dtype=np.float64), name="x")
        y = tf.Variable(tf.zeros(n, dtype=np.float64), name="y")
        z = tf.Variable(tf.zeros(n, dtype=np.float64), name="z")

        @TensorFlow((x, y), z)
        def cost(x, y, z):
            return tf.reduce_sum(x ** 2 + y + z ** 3)

        self.cost = cost


class TestVector(_backend_tests.TestVector):
    def setUp(self):
        super().setUp()

        n = self.n

        X = tf.Variable(tf.zeros(n, dtype=np.float64))

        @TensorFlow(X)
        def cost(X):
            return tf.exp(tf.reduce_sum(X ** 2))

        self.cost = cost


class TestMatrix(_backend_tests.TestMatrix):
    def setUp(self):
        super().setUp()

        m = self.m
        n = self.n

        X = tf.Variable(tf.zeros([m, n], dtype=np.float64))

        @TensorFlow(X)
        def cost(X):
            return tf.exp(tf.reduce_sum(X ** 2))

        self.cost = cost


class TestTensor3(unittest.TestCase):
    def setUp(self):
        n1 = self.n1 = 3
        n2 = self.n2 = 4
        n3 = self.n3 = 5

        self.X = X = tf.Variable(tf.zeros([n1, n2, n3]))

        @TensorFlow(X)
        def cost(X):
            return tf.exp(tf.reduce_sum(X ** 2))

        self.cost = cost

        Y = self.Y = rnd.randn(n1, n2, n3).astype(np.float32) * 1e-3
        A = self.A = rnd.randn(n1, n2, n3).astype(np.float32) * 1e-3

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
        np_testing.assert_allclose(self.correct_cost, self.cost(self.Y),
                                   rtol=1e-4)

    def test_grad(self):
        grad = self.cost.compute_gradient()
        np_testing.assert_allclose(self.correct_grad, grad(self.Y), rtol=1e-4)

    def test_hessian(self):
        hess = self.cost.compute_hessian()

        # Now test hess
        np_testing.assert_allclose(self.correct_hess, hess(self.Y, self.A),
                                   rtol=1e-4)


class TestMixed(unittest.TestCase):
    # Test autograd on a tuple containing vector, matrix and tensor3.
    def setUp(self):
        n1 = self.n1 = 3
        n2 = self.n2 = 4
        n3 = self.n3 = 5
        n4 = self.n4 = 6
        n5 = self.n5 = 7
        n6 = self.n6 = 8

        x = tf.Variable(tf.zeros([n1]))
        y = tf.Variable(tf.zeros([n2, n3]))
        z = tf.Variable(tf.zeros([n4, n5, n6]))

        @TensorFlow(x, y, z)
        def cost(x, y, z):
            return (tf.exp(tf.reduce_sum(x ** 2)) +
                    tf.exp(tf.reduce_sum(y ** 2)) +
                    tf.exp(tf.reduce_sum(z ** 2)))

        self.cost = cost
        self.arg = [x, y, z]

        self.y = y = (rnd.randn(n1).astype(np.float32) * 1e-3,
                      rnd.randn(n2, n3).astype(np.float32) * 1e-3,
                      rnd.randn(n4, n5, n6).astype(np.float32) * 1e-3)
        self.a = a = (rnd.randn(n1).astype(np.float32) * 1e-3,
                      rnd.randn(n2, n3).astype(np.float32) * 1e-3,
                      rnd.randn(n4, n5, n6).astype(np.float32) * 1e-3)

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

    def test_compile(self):
        np_testing.assert_allclose(self.correct_cost, self.cost(self.y))

    def test_grad(self):
        grad = self.cost.compute_gradient()
        for k in range(len(grad(self.y))):
            np_testing.assert_allclose(self.correct_grad[k], grad(self.y)[k],
                                       rtol=1e-4)

    def test_hessian(self):
        hess = self.cost.compute_hessian()

        # Now test hess
        for k in range(len(hess(self.y, self.a))):
            np_testing.assert_allclose(self.correct_hess[k],
                                       hess(self.y, self.a)[k], rtol=1e-4)


class TestUserProvidedSession(unittest.TestCase):
    def test_user_session(self):
        class MockSession:
            def run(*args, **kwargs):
                raise RuntimeError

        n = 10

        x = tf.Variable(tf.zeros(n, dtype=tf.float64), name="x")

        @TensorFlow(x, session=MockSession())
        def cost(x):
            return tf.reduce_sum(x)

        with self.assertRaises(RuntimeError):
            cost(rnd.randn(n))
