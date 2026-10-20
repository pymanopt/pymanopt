"""Tests for the line search methods."""

import numpy as np
import pytest

from pymanopt import Problem
from pymanopt.backends.numpy_backend import NumpyBackend
from pymanopt.manifolds import Euclidean, Sphere
from pymanopt.optimizers import SteepestDescent
from pymanopt.optimizers.line_search import ConstantLineSearcher


class TestConstantLineSearcher:
    @pytest.mark.parametrize("step_size", [1e-3, 0.5, 2])
    def test_search_returns_fixed_step(self, step_size):
        manifold = Sphere(4, backend=NumpyBackend())
        x = manifold.random_point()
        d = manifold.random_tangent_vector(x)

        def objective(point):
            raise AssertionError("The objective must not be evaluated.")

        line_searcher = ConstantLineSearcher(step_size=step_size)
        returned_step_size, newx = line_searcher.search(
            objective, manifold, x, d, f0=None, df0=None
        )

        assert returned_step_size == step_size
        assert isinstance(returned_step_size, float)
        np.testing.assert_allclose(newx, manifold.retraction(x, step_size * d))

    def test_steepest_descent_with_constant_step(self):
        # For f(x) = ||x||^2 / 2, a gradient step of size t maps x to
        # (1 - t) * x, so the iterates are known in closed form.
        manifold = Euclidean(3, backend=NumpyBackend())

        def cost(x):
            return float(0.5 * x @ x)

        def euclidean_gradient(x):
            return x

        problem = Problem(
            manifold, cost, euclidean_gradient=euclidean_gradient
        )
        step_size = 0.25
        max_iterations = 10
        initial_point = np.array([1.0, -2.0, 3.0])
        optimizer = SteepestDescent(
            line_searcher=ConstantLineSearcher(step_size=step_size),
            max_iterations=max_iterations,
            min_gradient_norm=0,
            verbosity=0,
        )
        result = optimizer.run(problem, initial_point=initial_point)

        assert result.iterations == max_iterations
        assert result.step_size == step_size
        np.testing.assert_allclose(
            result.point,
            (1 - step_size) ** max_iterations * initial_point,
        )
