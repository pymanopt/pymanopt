import abc
import time


class Solver(metaclass=abc.ABCMeta):
    """Abstract base class for Pymanopt solvers.

    Args:
        maxtime: Upper bound on the run time of a solver in seconds.
        maxiter: The maximum number of iterations to perform.
        mingradnorm: Termination threshold for the norm of the gradient.
        minstepsize: Termination threshold for the line search step size.
        maxcostevals: Maximum number of allowed cost function evaluations.
        logverbosity: Level of information logged by the solver while it
            operates: 0 is silent, 2 ist most verbose.
    """

    def __init__(
        self,
        maxtime=1000,
        maxiter=1000,
        mingradnorm=1e-6,
        minstepsize=1e-10,
        maxcostevals=5000,
        logverbosity=0,
    ):
        self._maxtime = maxtime
        self._maxiter = maxiter
        self._mingradnorm = mingradnorm
        self._minstepsize = minstepsize
        self._maxcostevals = maxcostevals
        self._logverbosity = logverbosity
        self._optlog = None

    def __str__(self):
        return type(self).__name__

    @abc.abstractmethod
    def solve(self, problem, x=None):
        """Run a solver on a given optimization problem.

        Solve the given :class:`pymanopt.core.problem.Problem` starting from
        ``x`` if provided or from a random initial guess if not.
        """

    def _check_stopping_criterion(
        self,
        time0,
        iter=-1,
        gradnorm=float("inf"),
        stepsize=float("inf"),
        costevals=-1,
    ):
        runtime = time.time() - time0
        reason = None
        if time.time() >= time0 + self._maxtime:
            reason = f"Terminated - max time reached after {iter} iterations."
        elif iter >= self._maxiter:
            reason = (
                "Terminated - max iterations reached after "
                f"{runtime:.2f} seconds."
            )
        elif gradnorm < self._mingradnorm:
            reason = (
                f"Terminated - min grad norm reached after {iter} "
                f"iterations, {runtime:.2f} seconds."
            )
        elif stepsize < self._minstepsize:
            reason = (
                f"Terminated - min stepsize reached after {iter} iterations, "
                f"{runtime:.2f} seconds."
            )
        elif costevals >= self._maxcostevals:
            reason = (
                "Terminated - max cost evals reached after "
                f"{runtime:.2f} seconds."
            )
        return reason

    def _start_optlog(self, solverparams=None, extraiterfields=None):
        if self._logverbosity <= 0:
            self._optlog = None
        else:
            self._optlog = {
                "solver": str(self),
                "stoppingcriteria": {
                    "maxtime": self._maxtime,
                    "maxiter": self._maxiter,
                    "mingradnorm": self._mingradnorm,
                    "minstepsize": self._minstepsize,
                    "maxcostevals": self._maxcostevals,
                },
                "solverparams": solverparams,
            }
        if self._logverbosity >= 2:
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
