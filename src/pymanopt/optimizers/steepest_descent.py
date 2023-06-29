import time
from copy import deepcopy

import numpy as np

from pymanopt.optimizers.line_search import BackTrackingLineSearcher
from pymanopt.optimizers.optimizer import Optimizer, OptimizerResult
from pymanopt.tools import printer


class SteepestDescent(Optimizer):
    """Riemannian steepest descent algorithm.

    Perform optimization using gradient descent with line search.
    This method first computes the gradient of the objective, and then
    optimizes by moving in the direction of steepest descent (which is the
    opposite direction to the gradient).

    Args:
        line_searcher: The line search method.
    """

    def __init__(self, line_searcher=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if line_searcher is None:
            self._line_searcher = BackTrackingLineSearcher()
        else:
            self._line_searcher = line_searcher
        self.line_searcher = None

    # Function to solve optimisation problem using steepest descent.
    def run(
        self, problem, *, initial_point=None, reuse_line_searcher=False
    ) -> OptimizerResult:
        """Run steepest descent algorithm.

        Args:
            problem: Pymanopt problem class instance exposing the cost function
                and the manifold to optimize over.
                The class must either
            initial_point: Initial point on the manifold.
                If no value is provided then a starting point will be randomly
                generated.
            reuse_line_searcher: Whether to reuse the previous line searcher.
                Allows to use information from a previous call to
                :meth:`solve`.

        Returns:
            Local minimum of the cost function, or the most recent iterate if
            algorithm terminated before convergence.
        """
        manifold = problem.manifold
        objective = problem.cost
        gradient = problem.riemannian_gradient

        if not reuse_line_searcher or self.line_searcher is None:
            self.line_searcher = deepcopy(self._line_searcher)
        line_searcher = self.line_searcher

        # If no starting point is specified, generate one at random.
        if initial_point is None:
            x = manifold.random_point()
        else:
            x = initial_point

        if self._verbosity >= 1:
            print("Optimizing...")
        if self._verbosity >= 2:
            iteration_format_length = int(np.log10(self._max_iterations)) + 1
            column_printer = printer.ColumnPrinter(
                columns=[
                    ("Iteration", f"{iteration_format_length}d"),
                    ("Cost", "+.16e"),
                    ("Gradient norm", ".8e"),
                ]
            )
        else:
            column_printer = printer.VoidPrinter()

        column_printer.print_header()

        self._initialize_log(
            optimizer_parameters={"line_searcher": line_searcher}
        )

        # Initialize iteration counter and timer
        iteration = 0
        start_time = time.time()

        while True:
            iteration += 1

            # Calculate new cost, grad and gradient_norm
            cost = objective(x)
            grad = gradient(x)
            gradient_norm = manifold.norm(x, grad)

            column_printer.print_row([iteration, cost, gradient_norm])

            self._add_log_entry(
                iteration=iteration,
                point=x,
                cost=cost,
                gradient_norm=gradient_norm,
            )

            # Descent direction is minus the gradient
            desc_dir = -grad

            # Perform line-search
            step_size, x = line_searcher.search(
                objective, manifold, x, desc_dir, cost, -(gradient_norm**2)
            )

            stopping_criterion = self._check_stopping_criterion(
                start_time=start_time,
                step_size=step_size,
                gradient_norm=gradient_norm,
                iteration=iteration,
            )

            if stopping_criterion:
                if self._verbosity >= 1:
                    print(stopping_criterion)
                    print("")
                break

        return self._return_result(
            start_time=start_time,
            point=x,
            cost=objective(x),
            iterations=iteration,
            stopping_criterion=stopping_criterion,
            cost_evaluations=iteration,
            step_size=step_size,
            gradient_norm=gradient_norm,
        )
