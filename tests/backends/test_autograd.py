import autograd.numpy as anp
import numpy as np
import pytest

import pymanopt
from pymanopt.backends.autograd_backend import AutogradBackend

from . import _backend_tests


class TestUnaryFunction(_backend_tests.TestUnaryFunction):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout, backend=AutogradBackend()
        )

        @pymanopt.function.autograd(self.manifold)
        def cost(x):
            return anp.sum(x**2)  # type: ignore

        self.cost = cost


class TestUnaryComplexFunction(_backend_tests.TestUnaryComplexFunction):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout,
            backend=AutogradBackend(np.complex128),
        )

        @pymanopt.function.autograd(self.manifold)
        def cost(x):
            return anp.real(anp.sum(x**2))  # type: ignore

        self.cost = cost


class TestUnaryVarargFunction(_backend_tests.TestUnaryFunction):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout, backend=AutogradBackend()
        )

        @pymanopt.function.autograd(self.manifold)
        def cost(*x):
            (x,) = x
            return anp.sum(x**2)  # type: ignore

        self.cost = cost


class TestNaryFunction(_backend_tests.TestNaryFunction):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout, backend=AutogradBackend()
        )

        @pymanopt.function.autograd(self.manifold)
        def cost(x, y):
            return x @ y

        self.cost = cost


class TestNaryVarargFunction(_backend_tests.TestNaryFunction):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout, backend=AutogradBackend()
        )

        @pymanopt.function.autograd(self.manifold)
        def cost(*args):
            return anp.dot(*args)  # type: ignore

        self.cost = cost


class TestNaryParameterGrouping(_backend_tests.TestNaryParameterGrouping):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout, backend=AutogradBackend()
        )

        @pymanopt.function.autograd(self.manifold)
        def cost(x, y, z):
            return anp.sum(x**2 + y + z**3)  # type: ignore

        self.cost = cost


class TestVector(_backend_tests.TestVector):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout, backend=AutogradBackend()
        )

        @pymanopt.function.autograd(self.manifold)
        def cost(X):
            return anp.exp(anp.sum(X**2))  # type: ignore

        self.cost = cost


class TestMatrix(_backend_tests.TestMatrix):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout, backend=AutogradBackend()
        )

        @pymanopt.function.autograd(self.manifold)
        def cost(X):
            return anp.exp(anp.sum(X**2))  # type: ignore

        self.cost = cost


class TestTensor3(_backend_tests.TestTensor3):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout, backend=AutogradBackend()
        )

        @pymanopt.function.autograd(self.manifold)
        def cost(X):
            return anp.exp(anp.sum(X**2))

        self.cost = cost


class TestMixed(_backend_tests.TestMixed):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout, backend=AutogradBackend()
        )

        @pymanopt.function.autograd(self.manifold)
        def cost(x, y, z):
            return (
                anp.exp(anp.sum(x**2))
                + anp.exp(anp.sum(y**2))
                + anp.exp(anp.sum(z**2))
            )

        self.cost = cost
