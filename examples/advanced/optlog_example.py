import pprint

import autograd.numpy as np

import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import SteepestDescent


if __name__ == "__main__":
    X = np.diag([3, 2, 1]) @ np.random.randn(3, 200)
    manifold = Stiefel(3, 2)

    @pymanopt.function.autograd(manifold)
    def cost(w):
        return np.sum(np.sum((X - w @ w.T @ X) ** 2))

    solver = SteepestDescent(verbosity=0, log_verbosity=2)
    problem = pymanopt.Problem(manifold, cost)
    wopt, log = solver.solve(problem)

    print("Optimization log:")
    pprint.pprint(log)
