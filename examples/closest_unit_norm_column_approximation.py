import numpy as np
import numpy.random as rnd
import numpy.linalg as la
import theano.tensor as T

from pymanopt import Problem
from pymanopt.manifolds import Oblique
from pymanopt.solvers import ConjugateGradient


def closest_unit_norm_column_approximation(A):
    """
    Returns the matrix with unit-norm columns that is closests to A w.r.t. the
    Frobenius norm.
    """
    m, n = A.shape

    manifold = Oblique(m, n)
    solver = ConjugateGradient()
    X = T.matrix()
    cost = 0.5 * T.sum((X - A) ** 2)

    problem = Problem(man=manifold, ad_cost=cost, ad_arg=X)
    return solver.solve(problem)


if __name__ == "__main__":
    # Generate random problem data.
    m = 5
    n = 8
    A = rnd.randn(m, n)

    # Calculate the actual solution by normalizing the columns of A.
    X = A / la.norm(A, axis=0)[np.newaxis, :]

    # Solve the problem with pymanopt.
    Xopt = closest_unit_norm_column_approximation(A)

    # Print information about the solution.
    print
    print "solution found: %s" % np.allclose(X, Xopt, rtol=1e-3)
    print "Frobenius-error: %f" % la.norm(X - Xopt)
