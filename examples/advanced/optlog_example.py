import pprint

import autograd.numpy as np

import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import SteepestDescent


if __name__ == "__main__":
    X = np.diag([3, 2, 1]).dot(np.random.randn(3, 200))

    @pymanopt.function.Autograd
    def cost(w):
        return np.sum(np.sum((X - np.dot(w, np.dot(w.T, X))) ** 2))

    solver = SteepestDescent(logverbosity=2)
    manifold = Stiefel(3, 2)
    problem = pymanopt.Problem(manifold, cost, verbosity=0)
    wopt, optlog = solver.solve(problem)

    print("Optimization log:")
    pprint.pprint(optlog)
