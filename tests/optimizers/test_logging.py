"""Tests for the per-iteration logs of SteepestDescent and ConjugateGradient."""

import math

import numpy as np
import pytest

from pymanopt import Problem
from pymanopt.backends.numpy_backend import NumpyBackend
from pymanopt.manifolds import Sphere
from pymanopt.optimizers import (
    ConjugateGradient,
    SteepestDescent,
    TrustRegions,
)

LINE_SEARCH_OPTIMIZERS = (SteepestDescent, ConjugateGradient)
BASE_KEYS = {"time", "iteration", "point", "cost"}
LINE_SEARCH_KEYS = BASE_KEYS | {"gradient_norm", "step_size"}


@pytest.fixture
def problem():
    """Minimize a diagonal quadratic form on the sphere."""
    diagonal = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    manifold = Sphere(5, backend=NumpyBackend())

    def cost(x):
        return float(0.5 * x @ (diagonal * x))

    def euclidean_gradient(x):
        return diagonal * x

    def euclidean_hessian(x, u):
        return diagonal * u

    return Problem(
        manifold,
        cost,
        euclidean_gradient=euclidean_gradient,
        euclidean_hessian=euclidean_hessian,
    )


@pytest.fixture
def initial_point():
    point = np.arange(1.0, 6.0)
    return point / np.linalg.norm(point)


def run(optimizer_class, problem, initial_point, **kwargs):
    kwargs.setdefault("max_iterations", 100)
    optimizer = optimizer_class(verbosity=0, **kwargs)
    return optimizer.run(problem, initial_point=initial_point)


@pytest.mark.parametrize("optimizer_class", LINE_SEARCH_OPTIMIZERS)
@pytest.mark.parametrize("max_iterations", [1, 5, 100])
def test_log_contains_initial_and_final_iterates(
    optimizer_class, problem, initial_point, max_iterations
):
    result = run(
        optimizer_class,
        problem,
        initial_point,
        log_verbosity=1,
        max_iterations=max_iterations,
    )
    log = result.log["iterations"]

    assert set(log) == LINE_SEARCH_KEYS
    num_entries = len(log["iteration"])
    for values in log.values():
        assert len(values) == num_entries
    assert log["iteration"] == list(range(num_entries))
    if optimizer_class is SteepestDescent:
        assert num_entries == result.iterations + 1

    # Iteration 0 is the initial point, which is reached without a step.
    np.testing.assert_array_equal(log["point"][0], initial_point)
    assert math.isnan(log["step_size"][0])
    assert all(step_size > 0 for step_size in log["step_size"][1:])

    # The last entry is the returned point.
    np.testing.assert_array_equal(log["point"][-1], result.point)
    assert log["cost"][-1] == result.cost
    assert log["gradient_norm"][-1] == result.gradient_norm

    assert np.all(np.diff(log["time"]) >= 0)
    manifold = problem.manifold
    for point, cost, gradient_norm in zip(
        log["point"], log["cost"], log["gradient_norm"]
    ):
        assert cost == pytest.approx(problem.cost(point), abs=1e-12)
        assert gradient_norm == pytest.approx(
            manifold.norm(point, problem.riemannian_gradient(point)),
            abs=1e-12,
        )


@pytest.mark.parametrize("optimizer_class", LINE_SEARCH_OPTIMIZERS)
def test_logging_preserves_trajectory(optimizer_class, problem, initial_point):
    result = run(optimizer_class, problem, initial_point, log_verbosity=1)
    result_without_log = run(optimizer_class, problem, initial_point)
    assert result_without_log.log["iterations"] is None
    np.testing.assert_array_equal(result.point, result_without_log.point)
    assert result.iterations == result_without_log.iterations
    reason = result.stopping_criterion.split(" after ")[0]
    reason_without_log = result_without_log.stopping_criterion.split(" after ")
    assert reason == reason_without_log[0]


def test_steepest_descent_gradient_norm_matches_returned_point(
    problem, initial_point
):
    result = run(
        SteepestDescent,
        problem,
        initial_point,
        min_gradient_norm=1e-4,
        max_iterations=1000,
    )
    assert "min grad norm" in result.stopping_criterion
    point = result.point
    manifold = problem.manifold
    gradient_norm = manifold.norm(point, problem.riemannian_gradient(point))
    assert result.gradient_norm == pytest.approx(gradient_norm, abs=1e-12)
    assert result.gradient_norm < 1e-4


@pytest.mark.parametrize(
    "optimizer_class", LINE_SEARCH_OPTIMIZERS + (TrustRegions,)
)
def test_log_callable(optimizer_class, problem, initial_point):
    calls = []

    def log_callable(*, point, cost, iteration):
        calls.append((point, cost, iteration))
        return {"first_coordinate": point[0], "num_calls": len(calls)}

    result = run(
        optimizer_class,
        problem,
        initial_point,
        log_verbosity=1,
        log_callable=log_callable,
    )
    log = result.log["iterations"]

    assert {"first_coordinate", "num_calls", "log_time"} <= set(log)
    num_entries = len(log["iteration"])
    for values in log.values():
        assert len(values) == num_entries
    assert len(calls) == num_entries
    assert log["num_calls"] == list(range(1, num_entries + 1))
    assert all(log_time >= 0 for log_time in log["log_time"])
    for index, (point, cost, iteration) in enumerate(calls):
        np.testing.assert_array_equal(point, log["point"][index])
        assert cost == log["cost"][index]
        assert iteration == log["iteration"][index]
        assert log["first_coordinate"][index] == point[0]


@pytest.mark.parametrize("optimizer_class", LINE_SEARCH_OPTIMIZERS)
def test_log_callable_is_not_called_without_logging(
    optimizer_class, problem, initial_point
):
    def log_callable(**kwargs):
        raise AssertionError("log_callable must not be called")

    result = run(
        optimizer_class, problem, initial_point, log_callable=log_callable
    )
    assert result.log["iterations"] is None


def test_log_callable_key_clash(problem, initial_point):
    def log_callable(*, point, cost, iteration):
        return {"cost": cost}

    with pytest.raises(ValueError, match="cost"):
        run(
            SteepestDescent,
            problem,
            initial_point,
            log_verbosity=1,
            log_callable=log_callable,
        )


def test_default_log_keys_are_unchanged(problem, initial_point):
    result = run(TrustRegions, problem, initial_point, log_verbosity=1)
    assert set(result.log["iterations"]) == BASE_KEYS | {
        "gradient_norm",
        "hess_vec_evals",
    }
