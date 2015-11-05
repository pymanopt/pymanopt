# Steepest descent (gradient descent) algorithm based on steepestdescent.m
# from the manopt MATLAB package.

from ..theano_functions import comp_diff
from ..linesearch import linesearch
from ..solver import Solver

class SteepestDescent(Solver):
    def __init__(self):
        # Set gradient descent default parameters
        # TODO: allow user to specify these
        self.minstepsize = 1e-10
        self.maxiter = 100
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
            
        cost = objective(x)
        grad = gradient(x)
        gradnorm = man.norm(x, grad)
        
        iter = 0
        
        print "iter\tcost\t\tgrad. norm"
        
        while iter < self.maxiter:
            print str(iter) + "\t" + str(cost) + "\t" + str(gradnorm)
            
            # Descent direction is minus the gradient
            desc_dir = -grad
            
            # Perform line-search
            step_size, x = self.searcher.search(objective, man, x, desc_dir,
                 cost, -gradnorm**2)
            
            # Calculate new cost, grad and gradnorm
            cost = objective(x)
            grad = gradient(x)
            gradnorm = man.norm(x, grad)
            iter = iter + 1
            
        print x