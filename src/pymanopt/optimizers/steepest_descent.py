import time
from copy import deepcopy
from math import log10, nan

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

    Notes:
        If logging is enabled (``log_verbosity >= 1``), the initial point is
        logged as iteration 0 and every subsequent entry records the iterate
        reached after a completed step, so the log contains
        ``result.iterations + 1`` entries and its last entry corresponds to
        the returned point. Besides the keys logged by every optimizer, each
        entry contains the ``"gradient_norm"`` and the ``"step_size"`` of the
        step that led to the logged point (``nan`` for iteration 0).
    """

    def __init__(self, line_searcher=None, **kwargs):
        super().__init__(**kwargs)

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
            iteration_format_length = int(log10(self._max_iterations)) + 1
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

        # Calculate cost, grad and gradient_norm at the initial point
        cost = objective(x)
        grad = gradient(x)
        gradient_norm = manifold.norm(x, grad)
        step_size = nan

        # Initialize iteration counter and timer
        iteration = 0
        start_time = time.time()

        # Log the initial point as iteration 0
        self._add_log_entry(
            iteration=iteration,
            point=x,
            cost=cost,
            gradient_norm=gradient_norm,
            step_size=step_size,
        )

        while True:
            iteration += 1

            column_printer.print_row(
                [iteration, float(cost), float(gradient_norm)]
            )

            # Descent direction is minus the gradient
            desc_dir = -grad

            # Perform line-search
            step_size, x = line_searcher.search(
                objective, manifold, x, desc_dir, cost, -(gradient_norm**2)
            )

            # Calculate cost, grad and gradient_norm at the new iterate
            cost = objective(x)
            grad = gradient(x)
            gradient_norm = manifold.norm(x, grad)

            # Log every completed step, including the last one
            self._add_log_entry(
                iteration=iteration,
                point=x,
                cost=cost,
                gradient_norm=gradient_norm,
                step_size=step_size,
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
            cost=cost,
            iterations=iteration,
            stopping_criterion=stopping_criterion,
            cost_evaluations=iteration,
            step_size=step_size,
            gradient_norm=gradient_norm,
        )
