# Steepest descent (gradient descent) algorithm based on steepestdescent.m
# from the manopt MATLAB package.

from .theano_functions import comp_diff
import linesearch
from .solver import Solver

class SteepestDescent(Solver):
    def __init__(self, max_iter=100, min_stepsize = 1e-6):
        # Set gradient descent default parameters
        # TODO: allow user to specify these
        self.minstepsize = min_stepsize
        self.maxiter = max_iter
        self.tolgradnorm = 1e-6
        self.searcher = linesearch.Linesearch()

    # Function to solve optimisation problem using steepest descent.
    def solve(self, obj, arg, man, x=None):
        # Compile the objective function and compute and compile its
        # gradient.
        print "Computing gradient and compiling..."
        objective = comp_diff.compile(obj, arg)
        gradient = comp_diff.gradient(obj, arg)

        # If no starting point is specified, generate one at random.
        if not x:
            x = man.rand()

        print "Optimizing..."
        iter = 0

        # print "iter\tcost\t\tgrad. norm"

        while iter < self.maxiter:
            # Calculate new cost, grad and gradnorm
            cost = objective(x)
            grad = man.egrad2rgrad(x, gradient(x))
            gradnorm = man.norm(x, grad)
            iter = iter + 1

            # print "%4d" % iter + "\t" + "%.5f" % cost + "\t" + "%.5f" % gradnorm

            # Descent direction is minus the gradient
            desc_dir = -grad

            # Perform line-search
            step_size, x = self.searcher.search(objective, man, x, desc_dir,
                 cost, -gradnorm**2)

            if step_size < self.minstepsize:
                print "Terminated — converged after %d iterations." % iter
                return x
        print "Terminated — max iterations."
        return x
