import autograd.numpy as np
import pytest

import pymanopt

from . import _backend_tests


class TestUnaryFunction(_backend_tests.TestUnaryFunction):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.autograd(self.manifold)
        def cost(x):
            return np.sum(x**2)

        self.cost = cost


class TestUnaryComplexFunction(_backend_tests.TestUnaryComplexFunction):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.autograd(self.manifold)
        def cost(x):
            return np.real(np.sum(x**2))

        self.cost = cost


class TestUnaryVarargFunction(_backend_tests.TestUnaryFunction):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.autograd(self.manifold)
        def cost(*x):
            (x,) = x
            return np.sum(x**2)

        self.cost = cost


class TestNaryFunction(_backend_tests.TestNaryFunction):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.autograd(self.manifold)
        def cost(x, y):
            return x @ y

        self.cost = cost


class TestNaryVarargFunction(_backend_tests.TestNaryFunction):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.autograd(self.manifold)
        def cost(*args):
            return np.dot(*args)

        self.cost = cost


class TestNaryParameterGrouping(_backend_tests.TestNaryParameterGrouping):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.autograd(self.manifold)
        def cost(x, y, z):
            return np.sum(x**2 + y + z**3)

        self.cost = cost


class TestVector(_backend_tests.TestVector):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.autograd(self.manifold)
        def cost(X):
            return np.exp(np.sum(X**2))

        self.cost = cost


class TestMatrix(_backend_tests.TestMatrix):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.autograd(self.manifold)
        def cost(X):
            return np.exp(np.sum(X**2))

        self.cost = cost


class TestTensor3(_backend_tests.TestTensor3):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.autograd(self.manifold)
        def cost(X):
            return np.exp(np.sum(X**2))

        self.cost = cost


class TestMixed(_backend_tests.TestMixed):
    @pytest.fixture(autouse=True)
    def setup(self):
        @pymanopt.function.autograd(self.manifold)
        def cost(x, y, z):
            return (
                np.exp(np.sum(x**2))
                + np.exp(np.sum(y**2))
                + np.exp(np.sum(z**2))
            )

        self.cost = cost
