import autograd.numpy as np

from pymanopt import Problem, AutogradFunction
from pymanopt.manifolds import Elliptope
from pymanopt.solvers import ConjugateGradient


def packing_on_the_sphere(n, k, epsilon):
    manifold = Elliptope(n, k)
    solver = ConjugateGradient(mingradnorm=1e-8, maxiter=1e5)

    @AutogradFunction
    def cost(X):
        Y = np.dot(X, X.T)
        # Shift the exponentials by the maximum value to reduce numerical
        # trouble due to possible overflows.
        s = np.triu(Y, 1).max()
        expY = np.exp((Y - s) / epsilon)
        # Zero out the diagonal
        expY -= np.diag(np.diag(expY))
        u = np.triu(expY, 1).sum()
        return s + epsilon * np.log(u)

    problem = Problem(manifold, cost)
    return solver.solve(problem)


if __name__ == "__main__":
    k = 3  # Dimension of the embedding space, i.e. R^k
    n = 24  # Points on the sphere
    # This value should be as close to 0 as affordable. If it is too close to
    # zero, optimization first becomes much slower, than simply doesn't work
    # anymore because of floating point overflow errors (NaN's and Inf's start
    # to appear). If it is too large, then log-sum-exp is a poor approximation
    # of the max function, and the spread will be less uniform. An okay value
    # seems to be 0.01 or 0.001 for example. Note that a better strategy than
    # using a small epsilon straightaway is to reduce epsilon bit by bit and to
    # warm-start subsequent optimization in that way. Trustregions will be more
    # appropriate for these fine tunings.
    epsilon = 0.0015

    # Evaluate the maximum inner product between any two points of X.
    Yopt = packing_on_the_sphere(n, k, epsilon)
    Xopt = Yopt.dot(Yopt.T)
    maxdot = np.triu(Xopt, 1).max()
    print(maxdot)
