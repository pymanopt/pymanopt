import time
from copy import deepcopy

import numpy as np

from pymanopt.solvers.linesearch import LineSearchBackTracking
from pymanopt.solvers.solver import Solver
from pymanopt.tools import printer


class SteepestDescent(Solver):
    """Riemannian steepest descent solver.

    Perform optimization using gradient descent with line search.
    This method first computes the gradient of the objective, and then
    optimizes by moving in the direction of steepest descent (which is the
    opposite direction to the gradient).

    Args:
        linesearch: The line search method.
    """

    def __init__(self, linesearch=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if linesearch is None:
            self._linesearch = LineSearchBackTracking()
        else:
            self._linesearch = linesearch
        self.linesearch = None

    # Function to solve optimisation problem using steepest descent.
    def solve(self, problem, x=None, reuselinesearch=False):
        """Run steepest descent algorithm.

        Args:
            problem: Pymanopt problem class instance exposing the cost function
                and the manifold to optimize over.
                The class must either
            x: Initial point on the manifold.
                If no value is provided then a starting point will be randomly
                generated.
            reuselinesearch: Whether to reuse the previous linesearch object.
                Allows to use information from a previous call to
                :meth:`solve`.

        Returns:
            Local minimum of the cost function, or the most recent iterate if
            algorithm terminated before convergence.
        """
        man = problem.manifold
        verbosity = problem.verbosity
        objective = problem.cost
        gradient = problem.grad

        if not reuselinesearch or self.linesearch is None:
            self.linesearch = deepcopy(self._linesearch)
        linesearch = self.linesearch

        # If no starting point is specified, generate one at random.
        if x is None:
            x = man.rand()

        if verbosity >= 1:
            print("Optimizing...")
        if verbosity >= 2:
            iter_format_length = int(np.log10(self._maxiter)) + 1
            column_printer = printer.ColumnPrinter(
                columns=[
                    ("Iteration", f"{iter_format_length}d"),
                    ("Cost", "+.16e"),
                    ("Gradient norm", ".8e"),
                ]
            )
        else:
            column_printer = printer.VoidPrinter()

        column_printer.print_header()

        self._start_optlog(
            extraiterfields=["gradnorm"],
            solverparams={"linesearcher": linesearch},
        )

        # Initialize iteration counter and timer
        iter = 0
        time0 = time.time()

        while True:
            # Calculate new cost, grad and gradnorm
            cost = objective(x)
            grad = gradient(x)
            gradnorm = man.norm(x, grad)
            iter = iter + 1

            column_printer.print_row([iter, cost, gradnorm])

            if self._logverbosity >= 2:
                self._append_optlog(iter, x, cost, gradnorm=gradnorm)

            # Descent direction is minus the gradient
            desc_dir = -grad

            # Perform line-search
            stepsize, x = linesearch.search(
                objective, man, x, desc_dir, cost, -(gradnorm ** 2)
            )

            stop_reason = self._check_stopping_criterion(
                time0, stepsize=stepsize, gradnorm=gradnorm, iter=iter
            )

            if stop_reason:
                if verbosity >= 1:
                    print(stop_reason)
                    print("")
                break

        if self._logverbosity <= 0:
            return x
        else:
            self._stop_optlog(
                x,
                objective(x),
                stop_reason,
                time0,
                stepsize=stepsize,
                gradnorm=gradnorm,
                iter=iter,
            )
            return x, self._optlog
