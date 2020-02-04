import os
import unittest

import numpy as np
import numpy.random as rnd
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

        @TensorFlow(x, y, z)
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

        X = tf.Variable(tf.zeros((m, n), dtype=np.float64))

        @TensorFlow(X)
        def cost(X):
            return tf.exp(tf.reduce_sum(X ** 2))

        self.cost = cost


class TestTensor3(_backend_tests.TestTensor3):
    def setUp(self):
        super().setUp()

        n1 = self.n1
        n2 = self.n2
        n3 = self.n3

        X = tf.Variable(tf.zeros([n1, n2, n3], dtype=np.float64))

        @TensorFlow(X)
        def cost(X):
            return tf.exp(tf.reduce_sum(X ** 2))

        self.cost = cost


class TestMixed(_backend_tests.TestMixed):
    def setUp(self):
        super().setUp()

        n1 = self.n1
        n2 = self.n2
        n3 = self.n3
        n4 = self.n4
        n5 = self.n5
        n6 = self.n6

        x = tf.Variable(tf.zeros(n1, dtype=np.float64))
        y = tf.Variable(tf.zeros([n2, n3], dtype=np.float64))
        z = tf.Variable(tf.zeros([n4, n5, n6], dtype=np.float64))

        @TensorFlow(x, y, z)
        def cost(x, y, z):
            return (tf.exp(tf.reduce_sum(x ** 2)) +
                    tf.exp(tf.reduce_sum(y ** 2)) +
                    tf.exp(tf.reduce_sum(z ** 2)))

        self.cost = cost


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
