import os

import tensorflow as tf

from pymanopt.function import TensorFlow
from . import _backend_tests

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class TestUnaryFunction(_backend_tests.TestUnaryFunction):
    def setUp(self):
        super().setUp()

        @TensorFlow
        def cost(x):
            return tf.reduce_sum(x ** 2)

        self.cost = cost


class TestNaryFunction(_backend_tests.TestNaryFunction):
    def setUp(self):
        super().setUp()

        @TensorFlow
        def cost(x, y):
            return tf.tensordot(x, y, axes=1)

        self.cost = cost


class TestNaryParameterGrouping(_backend_tests.TestNaryParameterGrouping):
    def setUp(self):
        super().setUp()

        @TensorFlow
        def cost(x, y, z):
            return tf.reduce_sum(x ** 2 + y + z ** 3)

        self.cost = cost


class TestVector(_backend_tests.TestVector):
    def setUp(self):
        super().setUp()

        @TensorFlow
        def cost(X):
            return tf.exp(tf.reduce_sum(X ** 2))

        self.cost = cost


class TestMatrix(_backend_tests.TestMatrix):
    def setUp(self):
        super().setUp()

        @TensorFlow
        def cost(X):
            return tf.exp(tf.reduce_sum(X ** 2))

        self.cost = cost


class TestTensor3(_backend_tests.TestTensor3):
    def setUp(self):
        super().setUp()

        @TensorFlow
        def cost(X):
            return tf.exp(tf.reduce_sum(X ** 2))

        self.cost = cost


class TestMixed(_backend_tests.TestMixed):
    def setUp(self):
        super().setUp()

        @TensorFlow
        def cost(x, y, z):
            return (tf.exp(tf.reduce_sum(x ** 2)) +
                    tf.exp(tf.reduce_sum(y ** 2)) +
                    tf.exp(tf.reduce_sum(z ** 2)))

        self.cost = cost
