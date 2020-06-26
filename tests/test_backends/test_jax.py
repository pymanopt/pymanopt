import jax.numpy as np

from pymanopt.function import Jax
from . import _backend_tests


class TestUnaryFunction(_backend_tests.TestUnaryFunction):
    def setUp(self):
        super().setUp()

        @Jax
        def cost(x):
            return np.sum(x ** 2)

        self.cost = cost


class TestNaryFunction(_backend_tests.TestNaryFunction):
    def setUp(self):
        super().setUp()

        @Jax
        def cost(x, y):
            return x @ y

        self.cost = cost


class TestNaryParameterGrouping(_backend_tests.TestNaryParameterGrouping):
    def setUp(self):
        super().setUp()

        @Jax
        def cost(x, y, z):
            return np.sum(x ** 2 + y + z ** 3)

        self.cost = cost


class TestVector(_backend_tests.TestVector):
    def setUp(self):
        super().setUp()

        @Jax
        def cost(X):
            return np.exp(np.sum(X ** 2))

        self.cost = cost


class TestMatrix(_backend_tests.TestMatrix):
    def setUp(self):
        super().setUp()

        @Jax
        def cost(X):
            return np.exp(np.sum(X ** 2))

        self.cost = cost


class TestTensor3(_backend_tests.TestTensor3):
    def setUp(self):
        super().setUp()

        @Jax
        def cost(X):
            return np.exp(np.sum(X ** 2))

        self.cost = cost


class TestMixed(_backend_tests.TestMixed):
    def setUp(self):
        super().setUp()

        @Jax
        def cost(x, y, z):
            return (np.exp(np.sum(x ** 2)) + np.exp(np.sum(y ** 2)) +
                    np.exp(np.sum(z ** 2)))

        self.cost = cost
