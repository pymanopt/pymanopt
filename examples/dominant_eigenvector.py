import numpy as np
import numpy.random as rnd
import numpy.linalg as la
import theano.tensor as T

from pymanopt import Problem
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
    x = T.matrix()
    cost = -x.T.dot(T.dot(A, x)).trace()

    problem = Problem(man=manifold, theano_cost=cost, theano_arg=x)
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
    # eigenvectors, of course, but for comparison we need to get rid of the
    # ambiguity.
    if np.sign(x[0]) != np.sign(xopt[0]):
        xopt = -xopt

    # Print information about the solution.
    print
    print "l2-norm of x: %f" % la.norm(x)
    print "l2-norm of xopt: %f" % la.norm(xopt)
    print "solution found: %s" % np.allclose(x, xopt, rtol=1e-3)
    print "l2-error: %f" % la.norm(x - xopt)
