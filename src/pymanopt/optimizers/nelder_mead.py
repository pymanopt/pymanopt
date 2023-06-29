import time

import numpy as np

import pymanopt
from pymanopt import tools
from pymanopt.optimizers.optimizer import Optimizer, OptimizerResult
from pymanopt.optimizers.steepest_descent import SteepestDescent


def compute_centroid(manifold, points):
    """Compute the centroid of `points` on the `manifold` as Karcher mean."""

    @pymanopt.function.numpy(manifold)
    def objective(*y):
        if manifold.num_values == 1:
            (y,) = y
        return sum([manifold.dist(y, point) ** 2 for point in points]) / 2

    @pymanopt.function.numpy(manifold)
    def gradient(*y):
        if manifold.num_values == 1:
            (y,) = y
        return -sum(
            [manifold.log(y, point) for point in points],
            manifold.zero_vector(y),
        )

    optimizer = SteepestDescent(max_iterations=15, verbosity=0)
    problem = pymanopt.Problem(
        manifold, objective, riemannian_gradient=gradient
    )
    return optimizer.run(problem).point


class NelderMead(Optimizer):
    """Nelder-Mead alglorithm.

    Perform optimization using the derivative-free Nelder-Mead minimization
    algorithm.

    Args:
        max_cost_evaluations: Maximum number of allowed cost function
            evaluations.
        max_iterations: Maximum number of allowed iterations.
        reflection: Determines how far to reflect away from the worst vertex:
            stretched (reflection > 1), compressed (0 < reflection < 1),
            or exact (reflection = 1).
        expansion: Factor by which to expand the reflected simplex.
        contraction: Factor by which to contract the reflected simplex.
    """

    def __init__(
        self,
        max_cost_evaluations=None,
        max_iterations=None,
        reflection=1,
        expansion=2,
        contraction=0.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._max_cost_evaluations = max_cost_evaluations
        self._max_iterations = max_iterations
        self._reflection = reflection
        self._expansion = expansion
        self._contraction = contraction

    def run(self, problem, *, initial_point=None) -> OptimizerResult:
        """Run Nelder-Mead algorithm.

        Args:
            problem: Pymanopt problem class instance exposing the cost function
                and the manifold to optimize over.
            initial_point: Initial point on the manifold.
                If no value is provided then a starting point will be randomly
                generated.

        Returns:
            Local minimum of the cost function, or the most recent iterate if
            algorithm terminated before convergence.
        """
        manifold = problem.manifold
        objective = problem.cost

        # Choose proper default algorithm parameters. We need to know about the
        # dimension of the manifold to limit the parameter range, so we have to
        # defer proper initialization until this point.
        dim = manifold.dim
        if self._max_cost_evaluations is None:
            self._max_cost_evaluations = max(1000, 2 * dim)
        if self._max_iterations is None:
            self._max_iterations = max(2000, 4 * dim)

        # If no initial simplex x is given by the user, generate one at random.
        num_points = int(dim + 1)
        if initial_point is None:
            x = [manifold.random_point() for _ in range(num_points)]
        elif (
            tools.is_sequence(initial_point)
            and len(initial_point) != num_points
        ):
            x = initial_point
        else:
            raise ValueError(
                "The initial simplex `initial_point` must be a sequence of "
                f"{num_points} points"
            )

        # Compute objective-related quantities for x, and setup a function
        # evaluations counter.
        costs = np.array([objective(xi) for xi in x])
        cost_evaluations = dim + 1

        # Sort simplex points by cost.
        order = np.argsort(costs)
        costs = costs[order]
        x = [x[i] for i in order]

        # Iteration counter (at any point, iteration is the number of fully
        # executed iterations so far).
        iteration = 0

        start_time = time.time()

        self._initialize_log()

        while True:
            iteration += 1

            if self._verbosity >= 2:
                print(
                    f"Cost evals: {cost_evaluations:7d}\t"
                    f"Best cost: {costs[0]:+.8e}"
                )

            # Sort simplex points by cost.
            order = np.argsort(costs)
            costs = costs[order]
            x = [x[i] for i in order]

            stopping_criterion = self._check_stopping_criterion(
                start_time=start_time,
                iteration=iteration,
                cost_evaluations=cost_evaluations,
            )
            if stopping_criterion:
                if self._verbosity >= 1:
                    print(stopping_criterion)
                    print("")
                break

            # Compute a centroid for the dim best points.
            xbar = compute_centroid(manifold, x[:-1])

            # Compute the direction for moving along the axis xbar - worst x.
            vec = manifold.log(xbar, x[-1])

            # Reflection step
            xr = manifold.retraction(xbar, -self._reflection * vec)
            costr = objective(xr)
            cost_evaluations += 1

            # If the reflected point is honorable, drop the worst point,
            # replace it by the reflected point and start a new iteration.
            if costr >= costs[0] and costr < costs[-2]:
                if self._verbosity >= 2:
                    print("Reflection")
                costs[-1] = costr
                x[-1] = xr
                continue

            # If the reflected point is better than the best point, expand.
            if costr < costs[0]:
                xe = manifold.retraction(xbar, -self._expansion * vec)
                coste = objective(xe)
                cost_evaluations += 1
                if coste < costr:
                    if self._verbosity >= 2:
                        print("Expansion")
                    costs[-1] = coste
                    x[-1] = xe
                    continue
                else:
                    if self._verbosity >= 2:
                        print("Reflection (failed expansion)")
                    costs[-1] = costr
                    x[-1] = xr
                    continue

            # If the reflected point is worse than the second to worst point,
            # contract.
            if costr >= costs[-2]:
                if costr < costs[-1]:
                    # do an outside contraction
                    xoc = manifold.retraction(xbar, -self._contraction * vec)
                    costoc = objective(xoc)
                    cost_evaluations += 1
                    if costoc <= costr:
                        if self._verbosity >= 2:
                            print("Outside contraction")
                        costs[-1] = costoc
                        x[-1] = xoc
                        continue
                else:
                    # do an inside contraction
                    xic = manifold.retraction(xbar, self._contraction * vec)
                    costic = objective(xic)
                    cost_evaluations += 1
                    if costic <= costs[-1]:
                        if self._verbosity >= 2:
                            print("Inside contraction")
                        costs[-1] = costic
                        x[-1] = xic
                        continue

            # If we get here, shrink the simplex around x[0].
            if self._verbosity >= 2:
                print("Shrinkage")
            x0 = x[0]
            for i in np.arange(1, dim + 1):
                x[i] = manifold.pair_mean(x0, x[i])
                costs[i] = objective(x[i])
            cost_evaluations += dim

        x = x[0]
        cost = objective(x)
        return self._return_result(
            start_time=start_time,
            point=x,
            cost=cost,
            iterations=iteration,
            stopping_criterion=stopping_criterion,
            cost_evaluations=cost_evaluations,
        )
