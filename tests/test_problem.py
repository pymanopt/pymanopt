import numpy as np
import pytest
import tensorflow as tf
from numpy import testing as np_testing

import pymanopt
from pymanopt.manifolds import Product, Sphere, Stiefel
from pymanopt.optimizers import TrustRegions


class TestProblem:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.n = 15
        self.manifold = Sphere(self.n)

        @pymanopt.function.tensorflow(self.manifold)
        def cost(X):
            return tf.exp(tf.reduce_sum(X**2))

        self.cost = cost

    def test_prepare(self):
        problem = pymanopt.Problem(self.manifold, self.cost)
        x = np.random.normal(size=self.n)
        np_testing.assert_allclose(
            2 * x * np.exp(np.sum(x**2)), problem.euclidean_gradient(x)
        )

    def test_attribute_override(self):
        problem = pymanopt.Problem(self.manifold, self.cost)
        with pytest.raises(AttributeError):
            problem.manifold = None

    def test_vararg_cost_on_product(self):
        shape = (3, 3)
        manifold = Product([Stiefel(*shape)] * 2)

        @pymanopt.function.tensorflow(manifold)
        def cost(*args):
            X, Y = args
            return tf.reduce_sum(X) + tf.reduce_sum(Y)

        problem = pymanopt.Problem(manifold, cost)
        optimizer = TrustRegions(max_iterations=1)
        Xopt, Yopt = optimizer.run(problem).point
        assert Xopt.shape == (3, 3)
        assert Yopt.shape == (3, 3)
