import theano.tensor as T
import numpy as np

from pymanopt import Problem
from pymanopt.manifolds import Euclidean
from pymanopt.solvers import TrustRegions


if __name__ == "__main__":
    # Cost function is the squared reconstruction error
    wT = T.matrix()
    yT = T.matrix()
    XT = T.matrix()
    cost = T.sum(T.sum((yT-wT.T.dot(XT))**2))

    # A solver that involves the hessian
    solver = TrustRegions()

    # R^3
    manifold = Euclidean(3, 1)

    # Create the problem with extra cost function arguments
    problem = Problem(man=manifold, cost=cost, arg=wT,
                      extra_args=[yT, XT], verbosity=0)

    # Solve 5 instances of the same type of problem for different data input
    for k in range(0, 5):
        # Generate random data
        X = np.random.randn(3, 200)
        Y = np.random.randn(1, 200)
        problem.extra_args = [Y, X]
        wopt = solver.solve(problem)
        print('Run {}'.format(k+1))
        print('Weights found by pymanopt (top) / '
              'closed form solution (bottom)')
        print(wopt.T)
        print(np.linalg.inv(X.dot(X.T)).dot(X).dot(Y.T).T)
        print('')
