import torch
import numpy as np

from pymanopt import Problem
from pymanopt.manifolds import Euclidean
from pymanopt.solvers import TrustRegions


if __name__ == "__main__":
    # Cost function is the squared reconstruction error
    X = torch.from_numpy(np.zeros((200, 3)))
    y = torch.from_numpy(np.zeros((200, 3)))

    def cost(w):
        return (y - X @ w).pow(2).sum()

    # A solver that involves the hessian
    solver = TrustRegions()

    # R^3
    manifold = Euclidean(3, 1)

    # Create the problem with extra cost function arguments
    problem = Problem(manifold=manifold, cost=cost, verbosity=0, arg=torch.Tensor())

    # Solve 5 instances of the same type of problem for different data input
    for k in range(0, 5):
        # Generate random data
        X = torch.from_numpy(np.random.randn(200, 3))
        y = torch.from_numpy(np.random.randn(200, 1))

        wopt = solver.solve(problem)
        print('Run {}'.format(k+1))
        print('Weights found by pymanopt (top) / '
              'closed form solution (bottom)')
        print(wopt)
        print(np.linalg.inv(X.numpy().T.dot(X)).dot(X.numpy().T).dot(y))
        print('')
