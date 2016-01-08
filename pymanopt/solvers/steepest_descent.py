"""
Module containing steepest descent (gradient descent) algorithm based on
steepestdescent.m from the manopt MATLAB package.
"""
import time
from pymanopt.solvers import linesearch
from pymanopt.solvers.solver import Solver
from pymanopt.solvers.theano_functions import comp_diff

class SteepestDescent(Solver):
    """
    Steepest descent solver class.
    Variable attributes (defaults in brackets):
        - maxiter (1000)
            Max number of iterations to run.
        - maxtime (1000)
            Max time (in seconds) to run.
        - mingradnorm (1e-6)
            Terminate if the norm of the gradient is below this.
        - minstepsize (1e-10)
            Terminate if linesearch returns a vector whose norm is below this.
        - verbosity (2)
            Level of information printed by the solver while it operates, 0 is
            silent, 2 is most information.

    Function attributes:
        - solve
            Perform optimization using steepest descent.
    """
    def __init__(self, mingradnorm=1e-6, maxiter=1000, maxtime=1000,
    minstepsize = 1e-10, verbosity=2):
        """
        Initialize variable attributes.
        """
        # Set gradient descent parameters
        self.mingradnorm = mingradnorm
        self.maxiter = maxiter
        self.maxtime = maxtime
        self.minstepsize = minstepsize
        self.searcher = linesearch.Linesearch()
        self.verbosity = verbosity

    # Function to solve optimisation problem using steepest descent.
    def solve(self, obj, arg, man, x=None):
        """
        Perform optimization using gradient descent with linesearch. Both obj
        and arg must be theano TensorVariable objects. This method first
        computes the gradient (derivative) of obj w.r.t. arg, and then optimizes
        by moving in the direction of steepest descent (which is the opposite
        direction to the gradient).
        Arguments:
            - obj
                Theano TensorVariable which is the scalar cost to be optimized,
                defined symbolically in terms of the TensorVariable arg.
            - arg
                Theano TensorVariable which is the matrix (or higher order
                tensor) being optimized over.
            - man
                Pymanopt manifold, which is the manifold to optimize over.
            - x=None
                Optional parameter. Starting point on the manifold. If none then
                a starting point will be randomly generated.
        Returns:
            - x
                Local minimum of obj, or if algorithm terminated before
                convergence x will be the point at which it terminated.
        """
        # Compile the objective function and compute and compile its
        # gradient.
        if self.verbosity >= 1: print "Computing gradient and compiling..."
        objective = comp_diff.compile(obj, arg)
        gradient = comp_diff.gradient(obj, arg)

        # If no starting point is specified, generate one at random.
        if x == None:
            x = man.rand()

        if self.verbosity >= 1: print "Optimizing..."
        # Initialize iteration counter and timer
        iter = 0
        time0 = time.time()

        if self.verbosity >= 2: print " iter\t\t   cost val\t    grad. norm"
        while True:
            # Calculate new cost, grad and gradnorm
            cost = objective(x)
            grad = man.egrad2rgrad(x, gradient(x))
            gradnorm = man.norm(x, grad)
            iter = iter + 1

            if self.verbosity >= 2:
                print "%5d\t%+.16e\t%.8e" %(iter,cost,gradnorm)

            # Descent direction is minus the gradient
            desc_dir = -grad

            # Perform line-search
            step_size, x = self.searcher.search(objective, man, x, desc_dir,
                 cost, -gradnorm**2)

            # Check stopping conditions
            if step_size < self.minstepsize:
                if self.verbosity >= 1:
                    print ("Terminated - min stepsize reached after %d "
                        "iterations, %.2f seconds."
                        % (iter, (time.time() - time0)))
                return x

            if gradnorm < self.mingradnorm:
                if self.verbosity >= 1:
                    print ("Terminated - min grad norm reached after %d "
                        "iterations, %.2f seconds."
                        % (iter, (time.time() - time0)))
                return x

            if iter >= self.maxiter:
                if self.verbosity >= 1:
                    print ("Terminated - max iterations reached after "
                    "%.2f seconds." % (time.time() - time0))
                return x

            if time.time() >= time0 + self.maxtime:
                if self.verbosity >= 1:
                    print ("Terminated - max time reached after %d iterations."
                    % iter)
                return x
