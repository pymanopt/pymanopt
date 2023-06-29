import autograd.numpy as np
import pytest

import pymanopt
import pymanopt.optimizers
from pymanopt.manifolds import Sphere


class TestOptimizers:
    @pytest.fixture(autouse=True)
    def setup(self):
        n = 32
        matrix = np.random.normal(size=(n, n))
        self.manifold = manifold = Sphere(n)
        self.max_iterations = 50

        @pymanopt.function.autograd(manifold)
        def cost(point):
            return -point.T @ matrix @ point

        self.problem = pymanopt.Problem(manifold, cost)

    @pytest.mark.parametrize("optimizer_name", pymanopt.optimizers.OPTIMIZERS)
    def test_optimizers(self, optimizer_name):
        optimizer = getattr(pymanopt.optimizers, optimizer_name)(
            max_iterations=self.max_iterations, verbosity=0
        )
        result = optimizer.run(self.problem)
        assert result.point.shape == self.manifold.random_point().shape
        assert result.time != 0
        assert result.iterations <= self.max_iterations
        assert isinstance(result.stopping_criterion, str)

    def test_optimization_log(self):
        optimizer = pymanopt.optimizers.ConjugateGradient(
            max_iterations=self.max_iterations, verbosity=0
        )
        result = optimizer.run(self.problem)
        assert (
            result.log["stopping_criteria"]["max_iterations"]
            == self.max_iterations
        )
        assert result.log["iterations"] is None

        optimizer = pymanopt.optimizers.ConjugateGradient(
            max_iterations=self.max_iterations, verbosity=0, log_verbosity=1
        )
        result = optimizer.run(self.problem)
        iterations = result.log["iterations"]
        assert iterations is not None
        assert len(iterations["cost"]) == result.iterations
