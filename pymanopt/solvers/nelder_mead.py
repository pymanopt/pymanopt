"""
Module containing a Nelder-Mead minimization alglorithm for derivative-free
minimization based on neldermead.m and centroid.m from the manopt MATLAB
package.
"""
import time

import numpy as np
import numpy.random as rnd

from pymanopt import Problem
from pymanopt.solvers.steepest_descent import SteepestDescent
from pymanopt.solvers.solver import Solver
from pymanopt.tools import theano_functions as tf


def compute_centroid(man, x):
    """
    Compute the centroid as Karcher mean of points x belonging to the manifold
    man.
    """
    n = len(x)

    def objective(y):  # weighted Frechet variance
        acc = 0
        for i in range(n):
            acc += man.dist(y, x[i]) ** 2
        return acc / 2

    def gradient(y):
        g = man.zerovec(y)
        for i in range(n):
            g -= man.log(y, x[i])
        return g

    # XXX: manopt runs a few TR iterations here. For us to do this, we either
    #      need to work out the Hessian of the Frechet variance by hand or
    #      implement approximations for the Hessian to use in the TR solver.
    #      This is because we cannot implement the Frechet variance with theano
    #      and compute the Hessian automatically due to dependency on the
    #      manifold-dependent distance function.
    solver = SteepestDescent(verbosity=0, maxiter=15)
    problem = Problem(man, cost=objective, grad=gradient)
    return solver.solve(problem)


class NelderMead(Solver):
    """
    Nelder-Mead method solver class.
    Variable attributes (defaults in brackets):
        - maxcostevals (max(5000, 2 * dim))
            Maximum number of allowed cost evaluations
        - maxiter (max(500, 4 * dim))
            Maximum number of allowed iterations
        - reflection (1)
            TODO
        - expansion (2)
            TODO
        - contraction (0.5)
            TODO
    """
    def __init__(self, maxcostevals=None, maxiter=None, reflection=1,
                 expansion=2, contraction=0.5, *args, **kwargs):
        super(NelderMead, self).__init__(*args, **kwargs)

        self._maxcostevals = maxcostevals
        self._maxiter = maxiter
        self._reflection = reflection
        self._expansion = expansion
        self._contraction = contraction

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
        Perform optimization using a Nelder-Mead minimization algorithm.
        Both obj and arg must be theano TensorVariable objects.
        Arguments:
            - problem
                Pymanopt problem setup using the Problem class, this must
                have a .man attribute specifying the manifold to optimize
                over, as well as a cost and enough information to compute
                the gradient of that cost.
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
            self._maxcostevals = max(1000, 2 * dim)
        if self._maxiter is None:
            self._maxiter = max(2000, 4 * dim)

        # If no initial simplex x is given by the user, generate one at random.
        if x is None:
            x = [man.rand() for i in range(dim + 1)]
        elif not hasattr(x, "__iter__"):
            raise ValueError("The initial simplex x must be iterable")
        else:
            # XXX: Is this necessary?
            if len(x) != dim + 1:
                print ("The simplex size was adapted to the dimension of the "
                       "manifold")
                x = x[:dim + 1]

        # Compute objective-related quantities for x, and setup a function
        # evaluations counter.
        costs = np.array([objective(xi) for xi in x])
        fy = list(costs)
        costevals = dim + 1

        # Sort simplex points by cost.
        order = np.argsort(costs)
        costs = costs[order]
        x = [x[i] for i in order]  # XXX: Probably inefficient

        # Iteration counter (at any point, iter is the number of fully executed
        # iterations so far).
        iter = 0

        time0 = time.time()

        while True:
            iter += 1

            if self._verbosity >= 2:
                print "Cost evals: %7d\tBest cost: %+.8e" % (
                    costevals, costs[0])

            # Sort simplex points by cost.
            order = np.argsort(costs)
            costs = costs[order]
            x = [x[i] for i in order]  # XXX: Probably inefficient

            stop_reason = self._check_stopping_criterion(
                iter, costevals, time0)
            if stop_reason:
                if self._verbosity >= 1:
                    print stop_reason
                    print
                break

            # Compute a centroid for the dim best points.
            xbar = compute_centroid(man, x[:-1])

            # Compute the direction for moving along the axis xbar - worst x.
            vec = man.log(xbar, x[-1])

            # Reflection step
            xr = man.exp(xbar, -self._reflection * vec)
            costr = objective(xr)
            costevals += 1

            # If the reflected point is honorable, drop the worst point,
            # replace it by the reflected point and start a new iteration.
            if costr >= costs[0] and costr < costs[-2]:
                print "Reflection"
                costs[-1] = costr
                x[-1] = xr
                continue

            # If the reflected point is better than the best point, expand.
            if costr < costs[0]:
                xe = man.exp(xbar, -self._expansion * vec)
                coste = objective(xe)
                costevals += 1
                if coste < costr:
                    print "Expansion"
                    costs[-1] = coste
                    x[-1] = xe
                    continue
                else:
                    print "Reflection (failed expansion)"
                    costs[-1] = costr
                    x[-1] = xr
                    continue

            # If the reflected point is worse than the second to worst point,
            # contract.
            if costr >= costs[-2]:
                if costr < costs[-1]:
                    # do an outside contraction
                    xoc = man.exp(xbar, -self._contraction * vec)
                    costoc = objective(xoc)
                    costevals += 1
                    if costoc <= costr:
                        print "Outside contraction"
                        costs[-1] = costoc
                        x[-1] = xoc
                        continue
                else:
                    # do an inside contraction
                    xic = man.exp(xbar, self._contraction * vec)
                    costic = objective(xic)
                    costevals += 1
                    if costic <= costs[-1]:
                        print "Inside contraction"
                        costs[-1] = costic
                        x[-1] = xic
                        continue

            # If we get here, shrink the simplex around x[0].
            print "Shrinkage"
            x0 = x[0]
            for i in np.arange(1, dim + 1):
                x[i] = man.pairmean(x0, x[i])
                costs[i] = objective(xi)
            costevals += dim

        return x[0]
