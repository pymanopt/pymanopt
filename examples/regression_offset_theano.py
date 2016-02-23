import numpy as np

import theano.tensor as T

from pymanopt import Problem
from pymanopt.solvers import TrustRegions
from pymanopt.manifolds import Euclidean, Product

if __name__ == "__main__":
    # Generate random data
    X = np.random.randn(3, 100)
    Y = X[0:1, :] - 2*X[1:2, :] + np.random.randn(1, 100) + 5

    # Cost function is the sqaured test error
    w = T.matrix()
    b = T.matrix()
    cost = T.sum((Y-w.T.dot(X)-b[0, 0])**2)

    # A solver that involves the hessian
    solver = TrustRegions()

    # R^3 x R^1
    manifold = Product([Euclidean(3, 1), Euclidean(1, 1)])

    # Solve the problem with pymanopt
    problem = Problem(man=manifold, cost=cost, arg=[w, b], verbosity=0)
    wopt = solver.solve(problem)

    print('Weights found by pymanopt (top) / '
          'closed form solution (bottom)')

    print(wopt[0].T)
    print(wopt[1])

    X1 = np.concatenate((X, np.ones((1, 100))), axis=0)
    wclosed = np.linalg.inv(X1.dot(X1.T)).dot(X1).dot(Y.T)
    print(wclosed[0:3].T)
    print(wclosed[3])
