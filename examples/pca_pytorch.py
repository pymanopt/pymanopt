import numpy as np
import torch

from pymanopt import Problem
from pymanopt.solvers import TrustRegions
from pymanopt.manifolds import Stiefel


if __name__ == "__main__":
    # Generate random data with highest variance in first 2 dimensions
    X = torch.from_numpy( np.diag([3, 2, 1]).dot(np.random.randn(3, 200)) )

    # Cost function is the squared reconstruction error
    def cost(w):
        return (X - w @ w.t() @ X).pow(2).sum()

    # A solver that involves the hessian
    solver = TrustRegions()

    # Projection matrices onto a two dimensional subspace
    manifold = Stiefel(3, 2)

    # Solve the problem with pymanopt
    problem = Problem(manifold=manifold, cost=cost, arg=torch.Tensor())
    wopt = solver.solve(problem)

    print('The following projection matrix was found to minimise '
          'the squared reconstruction error: ')
    print(wopt)
