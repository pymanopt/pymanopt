import enum
import time
from copy import deepcopy

import numpy as np

from pymanopt.optimizers.line_search import AdaptiveLineSearcher
from pymanopt.optimizers.optimizer import Optimizer, OptimizerResult
from pymanopt.tools import printer


class BetaRule(enum.Enum):
    FletcherReeves = enum.auto()
    PolakRibiere = enum.auto()
    HestenesStiefel = enum.auto()
    HagerZhang = enum.auto()


class ConjugateGradient(Optimizer):
    """Riemannian conjugate gradient method.

    Perform optimization using nonlinear conjugate gradient method with
    line_searcher.
    This method first computes the gradient of the cost function, and then
    optimizes by moving in a direction that is conjugate to all previous search
    directions.

    Args:
        beta_rule: Conjugate gradient beta rule used to construct the new
            search direction. Valid choices are ``{"FletcherReeves",
            "PolakRibiere", "HestenesStiefel", "HagerZhang"}``.
        orth_value: Parameter for Powell's restart strategy.
            An infinite value disables this strategy.
            See in code formula for the specific criterion used.
        line_searcher: The line search method.
    """

    def __init__(
        self,
        beta_rule: str = "HestenesStiefel",
        orth_value=np.inf,
        line_searcher=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        beta_rules = list(map(str, BetaRule.__members__))
        if beta_rule not in beta_rules:
            raise ValueError(
                f"Invalid beta rule {beta_rule}. Should be one of "
                f"{beta_rules}."
            )
        self._beta_rule = beta_rule
        self._orth_value = orth_value

        if line_searcher is None:
            self._line_searcher = AdaptiveLineSearcher()
        else:
            self._line_searcher = line_searcher
        self.line_searcher = None

    def run(
        self, problem, *, initial_point=None, reuse_line_searcher=False
    ) -> OptimizerResult:
        """Run CG method.

        Args:
            problem: Pymanopt problem class instance exposing the cost function
                and the manifold to optimize over.
                The class must either
            initial_point: Initial point on the manifold.
                If no value is provided then a starting point will be randomly
                generated.
            reuse_line_searcher: Whether to reuse the previous line searcher.
                Allows to use information from a previous call to
                :meth:`run`.

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

        # Calculate initial cost-related quantities
        cost = objective(x)
        grad = gradient(x)
        gradient_norm = manifold.norm(x, grad)
        Pgrad = problem.preconditioner(x, grad)
        gradPgrad = manifold.inner_product(x, grad, Pgrad)

        # Initial descent direction is the negative gradient
        desc_dir = -Pgrad

        self._initialize_log(
            optimizer_parameters={
                "beta_rule": self._beta_rule,
                "orth_value": self._orth_value,
                "line_searcher": line_searcher,
            },
        )

        # Initialize iteration counter and timer
        iteration = 0
        step_size = np.nan
        start_time = time.time()

        while True:
            iteration += 1

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

            # The line search algorithms require the directional derivative of
            # the cost at the current point x along the search direction.
            df0 = manifold.inner_product(x, grad, desc_dir)

            # If we didn't get a descent direction: restart, i.e., switch to
            # the negative gradient. Equivalent to resetting the CG direction
            # to a steepest descent step, which discards the past information.
            if df0 >= 0:
                # Or we switch to the negative gradient direction.
                if self._verbosity >= 3:
                    print(
                        "Conjugate gradient info: got an ascent direction "
                        f"(df0 = {df0:.2f}), reset to the (preconditioned) "
                        "steepest descent direction."
                    )
                # Reset to negative gradient: this discards the CG memory.
                desc_dir = -Pgrad
                df0 = -gradPgrad

            # Execute line search
            step_size, newx = line_searcher.search(
                objective, manifold, x, desc_dir, cost, df0
            )

            # Compute the new cost-related quantities for newx
            newcost = objective(newx)
            newgrad = gradient(newx)
            newgradient_norm = manifold.norm(newx, newgrad)
            Pnewgrad = problem.preconditioner(newx, newgrad)
            newgradPnewgrad = manifold.inner_product(newx, newgrad, Pnewgrad)

            # Apply the CG scheme to compute the next search direction
            oldgrad = manifold.transport(x, newx, grad)
            orth_grads = (
                manifold.inner_product(newx, oldgrad, Pnewgrad)
                / newgradPnewgrad
            )

            # Powell's restart strategy (see page 12 of Hager and Zhang's
            # survey on conjugate gradient methods, for example)
            if abs(orth_grads) >= self._orth_value:
                beta = 0
                desc_dir = -Pnewgrad
            else:
                desc_dir = manifold.transport(x, newx, desc_dir)

                # TODO(nkoep): Define closures for these in the constructor.
                if self._beta_rule == "FletcherReeves":
                    beta = newgradPnewgrad / gradPgrad
                elif self._beta_rule == "PolakRibiere":
                    diff = newgrad - oldgrad
                    ip_diff = manifold.inner_product(newx, Pnewgrad, diff)
                    beta = max(0, ip_diff / gradPgrad)
                elif self._beta_rule == "HestenesStiefel":
                    diff = newgrad - oldgrad
                    ip_diff = manifold.inner_product(newx, Pnewgrad, diff)
                    try:
                        beta = max(
                            0,
                            ip_diff
                            / manifold.inner_product(newx, diff, desc_dir),
                        )
                    except ZeroDivisionError:
                        beta = 1
                elif self._beta_rule == "HagerZhang":
                    diff = newgrad - oldgrad
                    Poldgrad = manifold.transport(x, newx, Pgrad)
                    Pdiff = Pnewgrad - Poldgrad
                    deno = manifold.inner_product(newx, diff, desc_dir)
                    numo = manifold.inner_product(newx, diff, Pnewgrad)
                    numo -= (
                        2
                        * manifold.inner_product(newx, diff, Pdiff)
                        * manifold.inner_product(newx, desc_dir, newgrad)
                        / deno
                    )
                    beta = numo / deno
                    # Robustness (see Hager-Zhang paper mentioned above)
                    desc_dir_norm = manifold.norm(newx, desc_dir)
                    eta_HZ = -1 / (desc_dir_norm * min(0.01, gradient_norm))
                    beta = max(beta, eta_HZ)
                else:
                    raise AssertionError("unreachable")

                desc_dir = -Pnewgrad + beta * desc_dir

            # Update the necessary variables for the next iteration.
            x = newx
            cost = newcost
            grad = newgrad
            Pgrad = Pnewgrad
            gradient_norm = newgradient_norm
            gradPgrad = newgradPnewgrad

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
