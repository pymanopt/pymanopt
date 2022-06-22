import time

import numpy as np

from pymanopt import tools
from pymanopt.optimizers.optimizer import Optimizer, OptimizerResult
from pymanopt.tools import printer


class ParticleSwarm(Optimizer):
    """Particle swarm optimization (PSO) method.

    Perform optimization using the derivative-free particle swarm optimization
    algorithm.

    Args:
        max_cost_evaluations: Maximum number of allowed cost evaluations.
        max_iterations: Maximum number of allowed iterations.
        population_size: Size of the considered swarm population.
        nostalgia: Quantifies performance relative to past performances.
        social: Quantifies performance relative to neighbors.
    """

    def __init__(
        self,
        max_cost_evaluations=None,
        max_iterations=None,
        population_size=None,
        nostalgia=1.4,
        social=1.4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._max_cost_evaluations = max_cost_evaluations
        self._max_iterations = max_iterations
        self._population_size = population_size
        self._nostalgia = nostalgia
        self._social = social

    def run(self, problem, *, initial_point=None) -> OptimizerResult:
        """Run PSO algorithm.

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
            self._max_cost_evaluations = max(5000, 2 * dim)
        if self._max_iterations is None:
            self._max_iterations = max(500, 4 * dim)
        if self._population_size is None:
            self._population_size = min(40, 10 * dim)

        # If no initial population x is given by the user, generate one at
        # random.
        if initial_point is None:
            x = [
                manifold.random_point()
                for i in range(int(self._population_size))
            ]
        elif tools.is_sequence(initial_point):
            if len(initial_point) != self._population_size:
                print(
                    "The population size was forced to the size of "
                    "the given initial population"
                )
                self._population_size = len(initial_point)
            x = initial_point
        else:
            raise ValueError("The initial population must be iterable")

        # Initialize personal best positions to the initial population.
        y = list(x)

        # Save a copy of the swarm at the previous iteration.
        xprev = list(x)

        # Initialize velocities for each particle.
        v = [manifold.random_tangent_vector(xi) for xi in x]

        # Compute cost for each particle xi.
        costs = np.array([objective(xi) for xi in x])
        fy = list(costs)
        cost_evaluations = self._population_size

        # Identify the best particle and store its cost/position.
        imin = costs.argmin()
        fbest = costs[imin]
        xbest = x[imin]

        if self._verbosity >= 2:
            iteration_format_length = int(np.log10(self._max_iterations)) + 1
            column_printer = printer.ColumnPrinter(
                columns=[
                    ("Iteration", f"{iteration_format_length}d"),
                    ("Cost evaluations", "7d"),
                    ("Best cost", "+.8e"),
                ]
            )
        else:
            column_printer = printer.VoidPrinter()

        column_printer.print_header()

        self._initialize_log()

        # Iteration counter (at any point, iteration is the number of fully
        # executed iterations so far).
        iteration = 0
        start_time = time.time()

        while True:
            iteration += 1

            column_printer.print_row([iteration, cost_evaluations, fbest])

            # FIXME(nkoep): This only makes sense once we provide a custom
            #               callback mechanism that actually checks 'xi'.
            #               Right now this loop is pointless since our default
            #               stopping criteria do not involve 'xi'.
            # Stop if any particle triggers a stopping criterion.
            for i, xi in enumerate(x):  # noqa
                stopping_criterion = self._check_stopping_criterion(
                    start_time=start_time,
                    iteration=iteration,
                    cost_evaluations=cost_evaluations,
                )
                if stopping_criterion is not None:
                    break
            if stopping_criterion:
                if self._verbosity >= 1:
                    print(stopping_criterion)
                    print("")
                break

            # Compute the inertia factor which we linearly decrease from 0.9 to
            # 0.4 from iteration = 0 to iteration = max_iterations.
            w = 0.4 + 0.5 * (1 - iteration / self._max_iterations)

            # Compute the velocities.
            for i, xi in enumerate(x):
                # Get the position and past best position of particle i.
                yi = y[i]

                # Get the previous position and velocity of particle i.
                xiprev = xprev[i]
                vi = v[i]

                # Compute the new velocity of particle i, composed of three
                # contributions.
                inertia = w * manifold.transport(xiprev, xi, vi)
                nostalgia = (
                    np.random.uniform()
                    * self._nostalgia
                    * manifold.log(xi, yi)
                )
                social = (
                    np.random.uniform()
                    * self._social
                    * manifold.log(xi, xbest)
                )

                v[i] = inertia + nostalgia + social

            # Backup the current swarm positions.
            xprev = list(x)

            # Update positions, personal bests and global best.
            for i, xi in enumerate(x):
                # Compute new position of particle i.
                x[i] = manifold.retraction(xi, v[i])
                # Compute new cost of particle i.
                fxi = objective(xi)

                # Update costs of the swarm.
                costs[i] = fxi
                # Update self-best if necessary.
                if fxi < fy[i]:
                    fy[i] = fxi
                    y[i] = xi
                    # Update global best if necessary.
                    if fy[i] < fbest:
                        fbest = fy[i]
                        xbest = xi
            cost_evaluations += self._population_size

        return self._return_result(
            start_time=start_time,
            point=xbest,
            cost=fbest,
            iterations=iteration,
            stopping_criterion=stopping_criterion,
            cost_evaluations=cost_evaluations,
        )
