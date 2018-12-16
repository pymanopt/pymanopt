import autograd.numpy as np

from pymanopt import Problem, Autograd
from pymanopt.solvers import SteepestDescent
from pymanopt.manifolds import Stiefel

import pprint

if __name__ == "__main__":
    # Generate random data with highest variance in first 2 dimensions
    X = np.diag([3, 2, 1]).dot(np.random.randn(3, 200))

    # Cost function is the squared reconstruction error
    @Autograd
    def cost(w):
        return np.sum(np.sum((X - np.dot(w, np.dot(w.T, X)))**2))

    solver = SteepestDescent(logverbosity=2)

    # Projection matrices onto a two dimensional subspace
    manifold = Stiefel(3, 2)

    # Solve the problem with pymanopt
    problem = Problem(manifold=manifold, cost=cost, verbosity=0)
    wopt, optlog = solver.solve(problem)

    print('And here comes the optlog:\n\r')
    pp = pprint.PrettyPrinter()
    pp.pprint(optlog)
