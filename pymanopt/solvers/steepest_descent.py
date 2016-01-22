"""
Module containing steepest descent (gradient descent) algorithm based on
steepestdescent.m from the manopt MATLAB package.
"""
import time

from pymanopt.tools import theano_functions as tf
from pymanopt.solvers import linesearch
from pymanopt.solvers.solver import Solver


class SteepestDescent(Solver):
    def __init__(self, *args, **kwargs):
        super(SteepestDescent, self).__init__(*args, **kwargs)

        self._searcher = linesearch.LineSearch()

    # Function to solve optimisation problem using steepest descent.
    def solve(self, problem, x=None):
        """
        Perform optimization using gradient descent with linesearch. Both obj
        and arg must be theano TensorVariable objects. This method first
        computes the gradient (derivative) of obj w.r.t. arg, and then
        optimizes by moving in the direction of steepest descent (which is the
        opposite direction to the gradient).
        Arguments:
            - problem
                Pymanopt problem setup using the Problem class, this must
                have a .man attribute specifying the manifold to optimize
                over, as well as a cost and enough information to compute
                the gradient of that cost.
            - x=None
                Optional parameter. Starting point on the manifold. If none
                then a starting point will be randomly generated.
        Returns:
            - x
                Local minimum of obj, or if algorithm terminated before
                convergence x will be the point at which it terminated.
        """
        man = problem.man
        # Compile the objective function and compute and compile its
        # gradient.
        if self._verbosity >= 1:
            print "Computing gradient and compiling..."
        problem.prepare(need_grad = True)

        cost = problem.cost
        grad = problem.grad

        # If no starting point is specified, generate one at random.
        if x is None:
            x = man.rand()

        if self._verbosity >= 1: print "Optimizing..."
        # Initialize iteration counter and timer
        iter = 0
        time0 = time.time()

        if self._verbosity >= 2: print " iter\t\t   cost val\t    grad. norm"
        while True:
            # Calculate new cost, grad and gradnorm
            cost = objective(x)
            grad = man.egrad2rgrad(x, gradient(x))
            gradnorm = man.norm(x, grad)
            iter = iter + 1

            if self._verbosity >= 2:
                print "%5d\t%+.16e\t%.8e" %(iter,cost,gradnorm)

            # Descent direction is minus the gradient
            desc_dir = -grad

            # Perform line-search
            step_size, x = self._searcher.search(objective, man, x, desc_dir,
                                                 cost, -gradnorm**2)

            # Check stopping conditions
            if step_size < self._minstepsize:
                if self._verbosity >= 1:
                    print ("Terminated - min stepsize reached after %d "
                        "iterations, %.2f seconds."
                        % (iter, (time.time() - time0)))
                return x

            if gradnorm < self._mingradnorm:
                if self._verbosity >= 1:
                    print ("Terminated - min grad norm reached after %d "
                        "iterations, %.2f seconds."
                        % (iter, (time.time() - time0)))
                return x

            if iter >= self._maxiter:
                if self._verbosity >= 1:
                    print ("Terminated - max iterations reached after "
                    "%.2f seconds." % (time.time() - time0))
                return x

            if time.time() >= time0 + self._maxtime:
                if self._verbosity >= 1:
                    print ("Terminated - max time reached after %d iterations."
                    % iter)
                return x
