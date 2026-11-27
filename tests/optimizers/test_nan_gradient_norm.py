"""Tests for the NaN gradient norm stopping criterion."""

import time

import numpy as np
import pytest

from pymanopt import Problem
from pymanopt.backends.numpy_backend import NumpyBackend
from pymanopt.manifolds import Euclidean
from pymanopt.optimizers import ConjugateGradient, SteepestDescent


def test_stopping_criterion_reports_nan_gradient_norm():
    optimizer = SteepestDescent(verbosity=0)
    reason = optimizer._check_stopping_criterion(
        start_time=time.time(), iteration=3, gradient_norm=np.nan
    )
    assert reason is not None
    assert reason.startswith("Terminated - gradient norm is NaN after 3")


def test_stopping_criterion_ignores_finite_gradient_norm():
    optimizer = SteepestDescent(verbosity=0)
    reason = optimizer._check_stopping_criterion(
        start_time=time.time(), iteration=3, gradient_norm=1.0, step_size=1.0
    )
    assert reason is None


@pytest.mark.parametrize(
    "optimizer_class", [SteepestDescent, ConjugateGradient]
)
def test_optimizers_stop_on_nan_gradient_norm(optimizer_class):
    manifold = Euclidean(3, backend=NumpyBackend())

    def cost(x):
        return float(x @ x)

    def euclidean_gradient(x):
        return np.full_like(x, np.nan)

    problem = Problem(manifold, cost, euclidean_gradient=euclidean_gradient)
    optimizer = optimizer_class(max_iterations=100, verbosity=0)
    result = optimizer.run(problem, initial_point=np.ones(3))

    assert "gradient norm is NaN" in result.stopping_criterion
    assert np.isnan(result.gradient_norm)
    assert result.iterations == 1
