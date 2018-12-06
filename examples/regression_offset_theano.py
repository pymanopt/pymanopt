import numpy as np

import theano.tensor as T

from pymanopt import Problem, TheanoFunction
from pymanopt.solvers import TrustRegions
from pymanopt.manifolds import Euclidean, Product

if __name__ == "__main__":
    # Generate random data
    X = np.random.randn(3, 100)
    Y = X[0:1, :] - 2*X[1:2, :] + np.random.randn(1, 100) + 5

    # Cost function is the squared test error
    w = T.matrix()
    b = T.matrix()
    cost = TheanoFunction(T.sum((Y-w.T.dot(X)-b[0, 0])**2), [w, b])

    # A solver that involves the Hessian
    solver = TrustRegions()

    # R^3 x R^1
    manifold = Product([Euclidean(3, 1), Euclidean(1, 1)])

    # Solve the problem with pymanopt
    problem = Problem(manifold=manifold, cost=cost, verbosity=0)
    wopt = solver.solve(problem)

    print('Weights found by pymanopt (top) / '
          'closed form solution (bottom)')

    print(wopt[0].T)
    print(wopt[1])
    print()

    X1 = np.concatenate((X, np.ones((1, 100))), axis=0)
    wclosed = np.linalg.inv(X1.dot(X1.T)).dot(X1).dot(Y.T)
    print(wclosed[0:3].T)
    print(wclosed[3])
