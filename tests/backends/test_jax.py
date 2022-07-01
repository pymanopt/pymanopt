import jax.numpy as jnp

import pymanopt

from . import _backend_tests


class TestUnaryFunction(_backend_tests.TestUnaryFunction):
    def setUp(self):
        super().setUp()

        @pymanopt.function.jax(self.manifold)
        def cost(x):
            return jnp.sum(x**2)

        self.cost = cost


class TestUnaryComplexFunction(_backend_tests.TestUnaryComplexFunction):
    def setUp(self):
        super().setUp()

        @pymanopt.function.jax(self.manifold)
        def cost(x):
            return jnp.real(jnp.sum(x**2))

        self.cost = cost


class TestUnaryVarargFunction(_backend_tests.TestUnaryFunction):
    def setUp(self):
        super().setUp()

        @pymanopt.function.jax(self.manifold)
        def cost(*x):
            (x,) = x
            return jnp.sum(x**2)

        self.cost = cost


class TestNaryFunction(_backend_tests.TestNaryFunction):
    def setUp(self):
        super().setUp()

        @pymanopt.function.jax(self.manifold)
        def cost(x, y):
            return x @ y

        self.cost = cost


class TestNaryVarargFunction(_backend_tests.TestNaryFunction):
    def setUp(self):
        super().setUp()

        @pymanopt.function.jax(self.manifold)
        def cost(*args):
            return jnp.dot(*args)

        self.cost = cost


class TestNaryParameterGrouping(_backend_tests.TestNaryParameterGrouping):
    def setUp(self):
        super().setUp()

        @pymanopt.function.jax(self.manifold)
        def cost(x, y, z):
            return jnp.sum(x**2 + y + z**3)

        self.cost = cost


class TestVector(_backend_tests.TestVector):
    def setUp(self):
        super().setUp()

        @pymanopt.function.jax(self.manifold)
        def cost(X):
            return jnp.exp(jnp.sum(X**2))

        self.cost = cost


class TestMatrix(_backend_tests.TestMatrix):
    def setUp(self):
        super().setUp()

        @pymanopt.function.jax(self.manifold)
        def cost(X):
            return jnp.exp(jnp.sum(X**2))

        self.cost = cost


class TestTensor3(_backend_tests.TestTensor3):
    def setUp(self):
        super().setUp()

        @pymanopt.function.jax(self.manifold)
        def cost(X):
            return jnp.exp(jnp.sum(X**2))

        self.cost = cost


class TestMixed(_backend_tests.TestMixed):
    def setUp(self):
        super().setUp()

        @pymanopt.function.jax(self.manifold)
        def cost(x, y, z):
            return (
                jnp.exp(jnp.sum(x**2))
                + jnp.exp(jnp.sum(y**2))
                + jnp.exp(jnp.sum(z**2))
            )

        self.cost = cost
