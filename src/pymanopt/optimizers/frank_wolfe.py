import time
from copy import deepcopy

import numpy as np

from pymanopt.optimizers.optimizer import Optimizer, OptimizerResult
from pymanopt.tools import printer

class FrankWolfe(Optimizer):
    """Riemannian Frank Wolfe method.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def sgnplus(self, D):
        Dsigns = np.sign(D)
        Dsigns[Dsigns == -1] = 0
        return Dsigns

    def step_direction(self, grad, X, L, U):
        svdres = np.linalg.svd(grad)
        Q = svdres.U
        D = svdres.S
        Qstar = svdres.Vh

        Lcap = Qstar @ X @ L @ X @ Q
        Ucap = Qstar @ X @ L @ X @ Q

        P = np.linalg.cholesky(Ucap - Lcap)
        Pstar = P.T.conj()

        return np.linalg.inv(X) @ Q @ (Pstar @ self.sgnplus(D) @ P + Lcap) @ Qstar @ np.linalg.inv(X)

    def run(
        self, problem, L, U, *, initial_point=None, reuse_line_searcher=False
    ) -> OptimizerResult:
        """Run FW method.

        Args:
            problem: Pymanopt problem class instance exposing the cost function
                and the manifold to optimize over.
                The class must either
            initial_point: Initial point on the manifold.
                If no value is provided then a starting point will be randomly
                generated.

        Returns:
            Local minimum of the cost function, or the most recent iterate if
            algorithm terminated before convergence.
        """
        manifold = problem.manifold
        objective = problem.cost
        gradient = problem.riemannian_gradient

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


        # Initialize iteration counter and timer.
        iteration = 0
        step_size = 1.0
        start_time = time.time()

        while True:

            cost = objective(x)
            grad = gradient(x)
            gradient_norm = manifold.norm(x, grad)

            Z = self.step_direction(grad, x, L, U)

            column_printer.print_row([iteration, cost, gradient_norm])

            self._add_log_entry(
                iteration=iteration,
                point=x,
                cost=cost,
                gradient_norm=gradient_norm,
            )

            stopping_criterion = self._check_stopping_criterion(
                start_time=start_time,
                gradient_norm=gradient_norm,
                iteration=iteration,
                step_size=step_size,
            )

            if stopping_criterion:
                if self._verbosity >= 1:
                    print(stopping_criterion)
                    print("")
                break

            newx = (1 - step_size) * x + step_size * Z

            # Compute the new cost-related quantities for newx
            newcost = objective(newx)
            newgrad = gradient(newx)
            newgradient_norm = manifold.norm(newx, newgrad)

            # Update the necessary variables for the next iteration.
            x = newx
            cost = newcost
            grad = newgrad
            gradient_norm = newgradient_norm
            step_size = 2 / (iteration + 2)

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
