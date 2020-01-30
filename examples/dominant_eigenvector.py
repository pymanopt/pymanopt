import numpy as np
import theano.tensor as T
from numpy import linalg as la, random as rnd

import pymanopt
from pymanopt.manifolds import Sphere
from pymanopt.solvers import ConjugateGradient


def dominant_eigenvector(A):
    """
    Returns the dominant eigenvector of the symmetric matrix A.

    Note: For the same A, this should yield the same as the dominant invariant
    subspace example with p = 1.
    """
    m, n = A.shape
    assert m == n, "matrix must be square"
    assert np.allclose(np.sum(A - A.T), 0), "matrix must be symmetric"

    manifold = Sphere(n)
    solver = ConjugateGradient(maxiter=500, minstepsize=1e-6)
    x = T.vector()

    @pymanopt.function.Theano(x)
    def cost(x):
        return -x.T.dot(T.dot(A, x))

    problem = pymanopt.Problem(manifold, cost)
    xopt = solver.solve(problem)
    return xopt.squeeze()


if __name__ == "__main__":
    # Generate random problem data.
    n = 128
    A = rnd.randn(n, n)
    A = 0.5 * (A + A.T)

    # Calculate the actual solution by a conventional eigenvalue decomposition.
    w, v = la.eig(A)
    x = v[:, np.argmax(w)]

    # Solve the problem with pymanopt.
    xopt = dominant_eigenvector(A)

    # Make sure both vectors have the same direction. Both are valid
    # eigenvectors, but for comparison we need to get rid of the sign
    # ambiguity.
    if np.sign(x[0]) != np.sign(xopt[0]):
        xopt = -xopt

    # Print information about the solution.
    print('')
    print("l2-norm of x: %f" % la.norm(x))
    print("l2-norm of xopt: %f" % la.norm(xopt))
    print("l2-error: %f" % la.norm(x - xopt))
