import autograd.numpy as np

from pymanopt import Problem, AutogradFunction
from pymanopt.manifolds import Euclidean
from pymanopt.solvers import TrustRegions


if __name__ == "__main__":
    # Cost function is the squared reconstruction error
    X = np.zeros((200, 3))
    y = np.zeros((200, 3))

    @AutogradFunction
    def cost(w):
        return np.sum((y - np.dot(X, w)) ** 2)
    # A solver that involves the hessian
    solver = TrustRegions()

    # R^3
    manifold = Euclidean(3, 1)

    # Create the problem with extra cost function arguments
    problem = Problem(manifold=manifold, cost=cost, verbosity=0)

    # Solve 5 instances of the same type of problem for different data input
    for k in range(0, 5):
        # Generate random data
        X = np.random.randn(200, 3)
        y = np.random.randn(200, 1)

        wopt = solver.solve(problem)
        print('Run {}'.format(k+1))
        print('Weights found by pymanopt (top) / '
              'closed form solution (bottom)')
        print(wopt)
        print(np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y))
        print('')
