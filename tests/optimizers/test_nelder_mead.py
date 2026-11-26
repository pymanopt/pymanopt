"""Regression tests for user-supplied NelderMead initial simplices."""

import numpy as np
import pytest

from pymanopt import Problem
from pymanopt.backends.numpy_backend import NumpyBackend
from pymanopt.manifolds import Sphere
from pymanopt.optimizers import NelderMead


def make_problem():
    manifold = Sphere(3, backend=NumpyBackend())
    target = np.array([0.0, 0.0, 1.0])

    def cost(point):
        return -float(target @ point)

    return manifold, Problem(manifold, cost)


def test_uses_initial_simplex_with_dim_plus_one_points():
    manifold, problem = make_problem()
    simplex = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.6, 0.8]),
        np.array([0.0, -1.0, 0.0]),
    ]
    assert len(simplex) == manifold.dim + 1

    # Stop before the first update so the result is the best supplied vertex.
    result = NelderMead(max_iterations=1, verbosity=0).run(
        problem, initial_point=simplex
    )

    np.testing.assert_allclose(result.point, simplex[1])
    assert result.cost == pytest.approx(-0.8)


@pytest.mark.parametrize("num_points", [1, 2, 4, 5])
def test_rejects_initial_simplex_of_wrong_size(num_points):
    manifold, problem = make_problem()
    simplex = [manifold.random_point() for _ in range(num_points)]

    with pytest.raises(ValueError, match="sequence of 3 points"):
        NelderMead(max_iterations=10, verbosity=0).run(
            problem, initial_point=simplex
        )


def test_rejects_single_point_as_initial_simplex():
    manifold, problem = make_problem()

    with pytest.raises(ValueError, match="sequence of 3 points"):
        NelderMead(max_iterations=10, verbosity=0).run(
            problem, initial_point=manifold.random_point()
        )
