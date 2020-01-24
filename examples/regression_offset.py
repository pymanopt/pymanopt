from numpy import linalg as la
import numpy as np

import pymanopt
from pymanopt.manifolds import Euclidean, Product
from pymanopt.solvers import ConjugateGradient


if __name__ == "__main__":
    # Generate random data
    X = np.random.randn(100, 3)
    Y = X[:, 0:1] - 2*X[:, 1:2] + np.random.randn(100, 1) + 5

    ones = np.ones((100, 1))

    # Cost function is the squared test error. We use the ones vector for
    # emphasis that the offset b is a scalar, not a vector.
    @pymanopt.function.Callable
    def cost(x):
        w, b = x
        return la.norm(Y - np.dot(X, w) - b * ones) ** 2

    @pymanopt.function.Callable
    def egrad(x):
        w, b = x
        egrad_w = -2 * np.dot(X.T, Y - np.dot(X, w) - b * ones)
        # Note that b is scalar, hence the sum, i.e., the inner product with
        # the all ones vector.
        egrad_b = -2 * np.sum((Y - np.dot(X, w) - b * ones))
        return (egrad_w, egrad_b)

    # A solver that involves the Hessian
    solver = ConjugateGradient()

    # R^3 x R^1
    manifold = Product([Euclidean(3, 1), Euclidean(1, 1)])

    # Solve the problem with pymanopt
    problem = pymanopt.Problem(manifold, cost, egrad=egrad, verbosity=0)
    wopt = solver.solve(problem)

    print('Weights found by pymanopt (top) / '
          'closed form solution (bottom)')

    print(wopt[0].T)
    print(wopt[1])
    print()

    X1 = np.concatenate((X, np.ones((100, 1))), axis=1)
    wclosed = la.pinv(X1).dot(Y)
    print(wclosed[:3].T)
    print(wclosed[3])
