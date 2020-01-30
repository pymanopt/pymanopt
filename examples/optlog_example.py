import pprint

import autograd.numpy as np

import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import SteepestDescent


if __name__ == "__main__":
    # Generate random data with highest variance in first 2 dimensions
    X = np.diag([3, 2, 1]).dot(np.random.randn(3, 200))

    # Cost function is the squared reconstruction error
    @pymanopt.function.Autograd
    def cost(w):
        return np.sum(np.sum((X - np.dot(w, np.dot(w.T, X))) ** 2))

    solver = SteepestDescent(logverbosity=2)

    # Projection matrices onto a two dimensional subspace
    manifold = Stiefel(3, 2)

    # Solve the problem with pymanopt
    problem = pymanopt.Problem(manifold, cost, verbosity=0)
    wopt, optlog = solver.solve(problem)

    print('And here comes the optlog:\n\r')
    pp = pprint.PrettyPrinter()
    pp.pprint(optlog)
