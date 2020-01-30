import autograd.numpy as np

import pymanopt
from pymanopt.manifolds import Euclidean
from pymanopt.solvers import TrustRegions


if __name__ == "__main__":
    # Generate random data
    X = np.random.randn(3, 200)
    Y = np.random.randint(-5, 5, (1, 200))

    # Cost function is the squared error
    @pymanopt.function.Autograd
    def cost(w):
        return np.sum(np.sum((Y - np.dot(w.T, X)) ** 2))

    # A solver that involves the hessian
    solver = TrustRegions()

    # R^3
    manifold = Euclidean(3, 1)

    # Solve the problem with pymanopt
    problem = pymanopt.Problem(manifold, cost)
    wopt = solver.solve(problem)

    print('The following regression weights were found to minimise the '
          'squared error:')
    print(wopt)

    print('The closed form solution to this regression problem is:')
    print(np.linalg.inv(X.dot(X.T)).dot(X).dot(Y.T))
