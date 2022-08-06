import time
from copy import deepcopy

import numpy as np

from pymanopt.optimizers.line_search import AdaptiveLineSearcher
from pymanopt.optimizers.optimizer import Optimizer, OptimizerResult
from pymanopt.tools import printer


def _beta_fletcher_reeves(
    *,
    manifold,
    x,
    newx,
    grad,
    newgrad,
    Pnewgrad,
    newgradPnewgrad,
    Pgrad,
    gradPgrad,
    gradient_norm,
    oldgrad,
    descent_direction,
):
    return newgradPnewgrad / gradPgrad


def _beta_polak_ribiere(
    *,
    manifold,
    x,
    newx,
    grad,
    newgrad,
    Pnewgrad,
    newgradPnewgrad,
    Pgrad,
    gradPgrad,
    gradient_norm,
    oldgrad,
    descent_direction,
):
    ip_diff = manifold.inner_product(newx, Pnewgrad, newgrad - oldgrad)
    return max(0, ip_diff / gradPgrad)


def _beta_hestenes_stiefel(
    *,
    manifold,
    x,
    newx,
    grad,
    newgrad,
    Pnewgrad,
    newgradPnewgrad,
    Pgrad,
    gradPgrad,
    gradient_norm,
    oldgrad,
    descent_direction,
):
    diff = newgrad - oldgrad
    try:
        beta = max(
            0,
            manifold.inner_product(newx, Pnewgrad, diff)
            / manifold.inner_product(newx, diff, descent_direction),
        )
    except ZeroDivisionError:
        beta = 1
    return beta


def _beta_hager_zhang(
    *,
    manifold,
    x,
    newx,
    grad,
    newgrad,
    Pnewgrad,
    newgradPnewgrad,
    Pgrad,
    gradPgrad,
    gradient_norm,
    oldgrad,
    descent_direction,
):
    diff = newgrad - oldgrad
    Poldgrad = manifold.transport(x, newx, Pgrad)
    Pdiff = Pnewgrad - Poldgrad
    denominator = manifold.inner_product(newx, diff, descent_direction)
    numerator = manifold.inner_product(newx, diff, Pnewgrad)
    numerator -= (
        2
        * manifold.inner_product(newx, diff, Pdiff)
        * manifold.inner_product(newx, descent_direction, newgrad)
        / denominator
    )
    beta = numerator / denominator
    descent_direction_norm = manifold.norm(newx, descent_direction)
    eta_HZ = -1 / (descent_direction_norm * min(0.01, gradient_norm))
    return max(beta, eta_HZ)


def _beta_liu_storey(
    *,
    manifold,
    x,
    newx,
    grad,
    newgrad,
    Pnewgrad,
    newgradPnewgrad,
    Pgrad,
    gradPgrad,
    gradient_norm,
    oldgrad,
    descent_direction,
):
    diff = newgrad - oldgrad
    ip_diff = manifold.inner_product(newx, Pnewgrad, diff)
    denominator = -manifold.inner_product(x, grad, descent_direction)
    beta_ls = ip_diff / denominator
    beta_cd = newgradPnewgrad / denominator
    return max(0, min(beta_ls, beta_cd))


BETA_RULES = {
    "FletcherReeves": _beta_fletcher_reeves,
    "HagerZhang": _beta_hager_zhang,
    "HestenesStiefel": _beta_hestenes_stiefel,
    "PolakRibiere": _beta_polak_ribiere,
    "LiuStorey": _beta_liu_storey,
}


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
        line_searcher: The line search method.

    Notes:
        See [HZ2006]_ for details about Powell's restart strategy.
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

        try:
            self._beta_update = BETA_RULES[beta_rule]
        except KeyError:
            raise ValueError(
                f"Invalid beta rule '{beta_rule}'. Should be one of "
                f"{list(BETA_RULES.keys())}."
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

        # Calculate initial cost-related quantities.
        cost = objective(x)
        grad = gradient(x)
        gradient_norm = manifold.norm(x, grad)
        Pgrad = problem.preconditioner(x, grad)
        gradPgrad = manifold.inner_product(x, grad, Pgrad)

        # Initial descent direction is the negative gradient.
        descent_direction = -Pgrad

        self._initialize_log(
            optimizer_parameters={
                "beta_rule": self._beta_rule,
                "orth_value": self._orth_value,
                "line_searcher": line_searcher,
            },
        )

        # Initialize iteration counter and timer.
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
            df0 = manifold.inner_product(x, grad, descent_direction)

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
                descent_direction = -Pgrad
                df0 = -gradPgrad

            # Execute line search
            step_size, newx = line_searcher.search(
                objective, manifold, x, descent_direction, cost, df0
            )

            # Compute the new cost-related quantities for newx
            newcost = objective(newx)
            newgrad = gradient(newx)
            newgradient_norm = manifold.norm(newx, newgrad)
            Pnewgrad = problem.preconditioner(newx, newgrad)
            newgradPnewgrad = manifold.inner_product(newx, newgrad, Pnewgrad)

            # Powell's restart strategy.
            oldgrad = manifold.transport(x, newx, grad)
            orth_grads = (
                manifold.inner_product(newx, oldgrad, Pnewgrad)
                / newgradPnewgrad
            )
            if abs(orth_grads) >= self._orth_value:
                beta = 0
                descent_direction = -Pnewgrad
            else:
                # Transport latest search direction to tangent space at new
                # estimate.
                descent_direction = manifold.transport(
                    x, newx, descent_direction
                )
                beta = self._beta_update(
                    manifold=manifold,
                    x=x,
                    newx=newx,
                    grad=grad,
                    newgrad=newgrad,
                    Pnewgrad=Pnewgrad,
                    newgradPnewgrad=newgradPnewgrad,
                    Pgrad=Pgrad,
                    gradPgrad=gradPgrad,
                    gradient_norm=gradient_norm,
                    oldgrad=oldgrad,
                    descent_direction=descent_direction,
                )
                descent_direction = -Pnewgrad + beta * descent_direction

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
