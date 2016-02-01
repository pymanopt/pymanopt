"""
Module containing the particle swarm optimization method based on pso.m from
the manopt MATLAB package.
"""
import time

import numpy as np
import numpy.random as rnd

from pymanopt.tools import theano_functions as tf
from pymanopt.solvers.solver import Solver


class ParticleSwarm(Solver):
    """
    Particle Swarm Optimization (PSO) solver class.
    Variable attributes (defaults in brackets):
        - maxcostevals (max(5000, 2 * dim))
            Maximum number of allowed cost evaluations
        - maxiter (max(500, 4 * dim))
            Maximum number of allowed iterations
        - populationsize (min(40, 10 * dim))
            Size of the considered swarm population
        - nostalgia (1.4)
            Quantifies performance relative to past performances
        - social (1.4)
            Quantifies performance relative to neighbors
    """
    def __init__(self, maxcostevals=None, maxiter=None, populationsize=None,
                 nostalgia=1.4, social=1.4, *args, **kwargs):
        super(ParticleSwarm, self).__init__(*args, **kwargs)

        self._maxcostevals = maxcostevals
        self._maxiter = maxiter
        self._populationsize = populationsize
        self._nostalgia = nostalgia
        self._social = social

    def _check_stopping_criterion(self, iter, costevals, time0):
        reason = None
        if iter >= self._maxiter:
            reason = ("Terminated - max iterations reached after "
                      "%.2f seconds." % (time.time() - time0))
        elif costevals >= self._maxcostevals:
            reason = ("Terminated - max cost evals reached after "
                      "%.2f seconds." % (time.time() - time0))
        elif time.time() >= time0 + self._maxtime:
            reason = ("Terminated - max time reached after %d iterations."
                      % iter)
        return reason

    def solve(self, problem, x=None):
        """
        Perform optimization using the particle swarm optimization algorithm.
        Both obj and arg must be theano TensorVariable objects.
        Arguments:
            - problem
                Pymanopt problem setup using the Problem class, this must
                have a .man attribute specifying the manifold to optimize
                over, as well as a cost (specified using a theano graph
                or as a python function).
            - x=None
                Optional parameter. Initial population of elements on the
                manifold. If None then an initial population will be randomly
                generated
        Returns:
            - x
                Local minimum of obj, or if algorithm terminated before
                convergence x will be the point at which it terminated
        """
        man = problem.man

        # Compile the objective function and compute and compile its
        # gradient.
        if self._verbosity >= 1:
            print "Compling objective function..."
        problem.prepare()

        objective = problem.cost

        # Choose proper default algorithm parameters. We need to know about the
        # dimension of the manifold to limit the parameter range, so we have to
        # defer proper initialization until this point.
        dim = man.dim
        if self._maxcostevals is None:
            self._maxcostevals = max(5000, 2 * dim)
        if self._maxiter is None:
            self._maxiter = max(500, 4 * dim)
        if self._populationsize is None:
            self._populationsize = min(40, 10 * dim)

        # If no initial population x is given by the user, generate one at
        # random.
        if x is None:
            x = [man.rand() for i in range(self._populationsize)]
        elif not hasattr(x, "__iter__"):
            raise ValueError("The initial population x must be iterable")
        else:
            if len(x) != self._populationsize:
                print ("The population size was forced to the size of the "
                       "given initial population")
                self._populationsize = len(x)

        # Initialize personal best positions to the initial population.
        y = list(x)

        # Save a copy of the swarm at the previous iteration.
        xprev = list(x)

        # Initialize velocities for each particle.
        v = [man.randvec(xi) for xi in x]

        # Compute cost for each particle xi.
        costs = np.array([objective(xi) for xi in x])
        fy = list(costs)
        costevals = self._populationsize

        # Identify the best particle and store its cost/position.
        imin = costs.argmin()
        fbest = costs[imin]
        xbest = x[imin]

        # Iteration counter (at any point, iter is the number of fully executed
        # iterations so far).
        iter = 0

        time0 = time.time()

        while True:
            iter += 1

            if self._verbosity >= 2:
                print "Cost evals: %7d\tBest cost: %+.8e" % (costevals, fbest)

            # Stop if any particle triggers a stopping criterion.
            for i, xi in enumerate(x):
                stop_reason = self._check_stopping_criterion(
                    iter, costevals, time0)
                if stop_reason is not None:
                    break
            if stop_reason:
                if self._verbosity >= 1:
                    print stop_reason
                    print
                break

            # Compute the inertia factor which we linearly decrease from 0.9 to
            # 0.4 from iter = 0 to iter = maxiter.
            w = 0.4 + 0.5 * (1 - iter / self._maxiter)

            # Compute the velocities.
            for i, xi in enumerate(x):
                # Get the position and past best position of particle i.
                yi = y[i]

                # Get the previous position and velocity of particle i.
                xiprev = xprev[i]
                vi = v[i]

                # Compute the new velocity of particle i, composed of three
                # contributions.
                inertia = w * man.transp(xiprev, xi, vi)
                nostalgia = rnd.rand() * self._nostalgia * man.log(xi, yi)
                social = rnd.rand() * self._social * man.log(xi, xbest)

                v[i] = inertia + nostalgia + social

            # Backup the current swarm positions.
            xprev = list(x)

            # Update positions, personal bests and global best.
            for i, xi in enumerate(x):
                # Compute new position of particle i.
                x[i] = man.retr(xi, v[i])
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
            costevals += self._populationsize

        return xbest
