import abc
import collections
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class OptimizerResult:
    point: Any
    cost: float
    iterations: int
    stopping_criterion: str
    time: float
    cost_evaluations: Optional[int] = None
    step_size: Optional[float] = None
    gradient_norm: Optional[float] = None
    log: Optional[Dict] = None


class Optimizer(metaclass=abc.ABCMeta):
    """Abstract base class for Pymanopt optimizers.

    Args:
        max_time: Upper bound on the run time of an optimizer in seconds.
        max_iterations: The maximum number of iterations to perform.
        min_gradient_norm: Termination threshold for the norm of the
            (Riemannian) gradient.
        min_step_size: Termination threshold for the line search step
            size.
        max_cost_evaluations: Maximum number of allowed cost function
            evaluations.
        verbosity: Level of information printed by the optimizer while it
            operates: 0 is silent, 2 is most verbose.
        log_verbosity: Level of information logged by the optimizer while it
            operates: 0 logs nothing, 1 logs information for each iteration.
    """

    def __init__(
        self,
        max_time: float = 1000,
        max_iterations: int = 1000,
        min_gradient_norm: float = 1e-6,
        min_step_size: float = 1e-10,
        max_cost_evaluations: int = 5000,
        verbosity: int = 2,
        log_verbosity: int = 0,
    ):
        self._max_time = max_time
        self._max_iterations = max_iterations
        self._min_gradient_norm = min_gradient_norm
        self._min_step_size = min_step_size
        self._max_cost_evaluations = max_cost_evaluations
        self._verbosity = verbosity
        self._log_verbosity = log_verbosity

        self._log = None

    def __str__(self):
        return type(self).__name__

    @abc.abstractmethod
    def run(self, problem, *, initial_point=None, **kwargs) -> OptimizerResult:
        """Run an optimizer on a given optimization problem.

        Args:
            problem: Pymanopt problem class instance exposing the cost function
                and the manifold to optimize over.
            initial_point: Initial point on the manifold.
                If no value is provided then a starting point will be randomly
                generated.
            *args: Potential optimizer-specific positional arguments.
            **kwargs: Potential optimizer-specific keyword arguments.

        Returns:
            The optimization result.
        """

    def _return_result(self, *, start_time, **kwargs) -> OptimizerResult:
        return OptimizerResult(
            time=time.time() - start_time, log=self._log, **kwargs
        )

    def _check_stopping_criterion(
        self,
        *,
        start_time,
        iteration=-1,
        gradient_norm=np.inf,
        step_size=np.inf,
        cost_evaluations=-1,
    ):
        run_time = time.time() - start_time
        reason = None
        if time.time() >= start_time + self._max_time:
            reason = (
                f"Terminated - max time reached after {iteration} iterations."
            )
        elif iteration >= self._max_iterations:
            reason = (
                "Terminated - max iterations reached after "
                f"{run_time:.2f} seconds."
            )
        elif gradient_norm < self._min_gradient_norm:
            reason = (
                f"Terminated - min grad norm reached after {iteration} "
                f"iterations, {run_time:.2f} seconds."
            )
        elif step_size < self._min_step_size:
            reason = (
                f"Terminated - min step_size reached after {iteration} "
                f"iterations, {run_time:.2f} seconds."
            )
        elif cost_evaluations >= self._max_cost_evaluations:
            reason = (
                "Terminated - max cost evals reached after "
                f"{run_time:.2f} seconds."
            )
        return reason

    def _initialize_log(self, *, optimizer_parameters=None):
        self._log = {
            "optimizer": str(self),
            "stopping_criteria": {
                "max_time": self._max_time,
                "max_iterations": self._max_iterations,
                "min_gradient_norm": self._min_gradient_norm,
                "min_step_size": self._min_step_size,
                "max_cost_evaluations": self._max_cost_evaluations,
            },
            "optimizer_parameters": optimizer_parameters,
            "iterations": collections.defaultdict(list)
            if self._log_verbosity >= 1
            else None,
        }

    def _add_log_entry(self, *, iteration, point, cost, **kwargs):
        if self._log_verbosity <= 0:
            return
        self._log["iterations"]["time"].append(time.time())
        self._log["iterations"]["iteration"].append(iteration)
        self._log["iterations"]["point"].append(point)
        self._log["iterations"]["cost"].append(cost)
        for key, value in kwargs.items():
            self._log["iterations"][key].append(value)
