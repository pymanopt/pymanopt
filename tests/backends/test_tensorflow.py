import pytest
import tensorflow as tf

import pymanopt

from . import _backend_tests


class TestUnaryFunction(_backend_tests.TestUnaryFunction):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.tensorflow(self.manifold)
        def cost(x):
            return tf.reduce_sum(x**2)

        self.cost = cost


class TestUnaryComplexFunction(_backend_tests.TestUnaryComplexFunction):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.tensorflow(self.manifold)
        def cost(x):
            return tf.math.real(tf.reduce_sum(x**2))

        self.cost = cost


class TestUnaryVarargFunction(_backend_tests.TestUnaryFunction):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.tensorflow(self.manifold)
        def cost(*x):
            (x,) = x
            return tf.reduce_sum(x**2)

        self.cost = cost


class TestNaryFunction(_backend_tests.TestNaryFunction):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.tensorflow(self.manifold)
        def cost(x, y):
            return tf.tensordot(x, y, axes=1)

        self.cost = cost


class TestNaryVarargFunction(_backend_tests.TestNaryFunction):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.tensorflow(self.manifold)
        def cost(*args):
            return tf.tensordot(*args, axes=1)

        self.cost = cost


class TestNaryParameterGrouping(_backend_tests.TestNaryParameterGrouping):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.tensorflow(self.manifold)
        def cost(x, y, z):
            return tf.reduce_sum(x**2 + y + z**3)

        self.cost = cost


class TestVector(_backend_tests.TestVector):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.tensorflow(self.manifold)
        def cost(X):
            return tf.exp(tf.reduce_sum(X**2))

        self.cost = cost


class TestMatrix(_backend_tests.TestMatrix):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.tensorflow(self.manifold)
        def cost(X):
            return tf.exp(tf.reduce_sum(X**2))

        self.cost = cost


class TestTensor3(_backend_tests.TestTensor3):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.tensorflow(self.manifold)
        def cost(X):
            return tf.exp(tf.reduce_sum(X**2))

        self.cost = cost


class TestMixed(_backend_tests.TestMixed):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.tensorflow(self.manifold)
        def cost(x, y, z):
            return (
                tf.exp(tf.reduce_sum(x**2))
                + tf.exp(tf.reduce_sum(y**2))
                + tf.exp(tf.reduce_sum(z**2))
            )

        self.cost = cost
