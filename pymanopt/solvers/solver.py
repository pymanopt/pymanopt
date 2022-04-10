import abc
import time


class Solver(metaclass=abc.ABCMeta):
    """Abstract base class for Pymanopt solvers.

    Args:
        max_time: Upper bound on the run time of a solver in seconds.
        max_iterations: The maximum number of iterations to perform.
        min_gradient_norm: Termination threshold for the norm of the
            gradient.
        min_step_size: Termination threshold for the line search step
            size.
        max_cost_evaluations: Maximum number of allowed cost function
            evaluations.
        log_verbosity: Level of information logged by the solver while it
            operates: 0 is silent, 2 is most verbose.
    """

    def __init__(
        self,
        max_time=1000,
        max_iterations=1000,
        min_gradient_norm=1e-6,
        min_step_size=1e-10,
        max_cost_evaluations=5000,
        log_verbosity=0,
    ):
        self._max_time = max_time
        self._max_iterations = max_iterations
        self._min_gradient_norm = min_gradient_norm
        self._min_step_size = min_step_size
        self._max_cost_evaluations = max_cost_evaluations
        self._log_verbosity = log_verbosity
        self._optlog = None

    def __str__(self):
        return type(self).__name__

    @abc.abstractmethod
    def solve(self, problem, initial_point=None, *args, **kwargs):
        """Run a solver on a given optimization problem.

        Args:
            problem: Pymanopt problem class instance exposing the cost function
                and the manifold to optimize over.
                The class must either
            initial_point: Initial point on the manifold.
                If no value is provided then a starting point will be randomly
                generated.
            *args: Potential solver-specific positional arguments.
            **kwargs: Potential solver-specific keyword arguments.
        """

    def _check_stopping_criterion(
        self,
        time0,
        iter=-1,
        gradnorm=float("inf"),
        stepsize=float("inf"),
        costevals=-1,
    ):
        run_time = time.time() - time0
        reason = None
        if time.time() >= time0 + self._max_time:
            reason = f"Terminated - max time reached after {iter} iterations."
        elif iter >= self._max_iterations:
            reason = (
                "Terminated - max iterations reached after "
                f"{run_time:.2f} seconds."
            )
        elif gradnorm < self._min_gradient_norm:
            reason = (
                f"Terminated - min grad norm reached after {iter} "
                f"iterations, {run_time:.2f} seconds."
            )
        elif stepsize < self._min_step_size:
            reason = (
                f"Terminated - min stepsize reached after {iter} iterations, "
                f"{run_time:.2f} seconds."
            )
        elif costevals >= self._max_cost_evaluations:
            reason = (
                "Terminated - max cost evals reached after "
                f"{run_time:.2f} seconds."
            )
        return reason

    def _start_optlog(self, solverparams=None, extraiterfields=None):
        if self._log_verbosity <= 0:
            self._optlog = None
        else:
            self._optlog = {
                "solver": str(self),
                "stoppingcriteria": {
                    "max_time": self._max_time,
                    "max_iterations": self._max_iterations,
                    "min_gradient_norm": self._min_gradient_norm,
                    "min_step_size": self._min_step_size,
                    "max_cost_evaluations": self._max_cost_evaluations,
                },
                "solverparams": solverparams,
            }
        if self._log_verbosity >= 2:
            if extraiterfields:
                self._optlog["iterations"] = {
                    "iteration": [],
                    "time": [],
                    "x": [],
                    "f(x)": [],
                }
                for field in extraiterfields:
                    self._optlog["iterations"][field] = []

    def _append_optlog(self, iteration, x, fx, **kwargs):
        # In case not every iteration is being logged
        self._optlog["iterations"]["iteration"].append(iteration)
        self._optlog["iterations"]["time"].append(time.time())
        self._optlog["iterations"]["x"].append(x)
        self._optlog["iterations"]["f(x)"].append(fx)
        for key in kwargs:
            self._optlog["iterations"][key].append(kwargs[key])

    def _stop_optlog(
        self,
        x,
        objective,
        stop_reason,
        time0,
        stepsize=float("inf"),
        gradnorm=float("inf"),
        iter=-1,
        costevals=-1,
    ):
        self._optlog["stoppingreason"] = stop_reason
        self._optlog["final_values"] = {
            "x": x,
            "f(x)": objective,
            "time": time.time() - time0,
        }
        if stepsize is not float("inf"):
            self._optlog["final_values"]["stepsize"] = stepsize
        if gradnorm is not float("inf"):
            self._optlog["final_values"]["gradnorm"] = gradnorm
        if iter != -1:
            self._optlog["final_values"]["iterations"] = iter
        if costevals != -1:
            self._optlog["final_values"]["costevals"] = costevals
