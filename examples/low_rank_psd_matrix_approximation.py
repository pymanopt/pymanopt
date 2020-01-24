import numpy as np
import theano.tensor as T
from numpy import linalg as la, random as rnd

import pymanopt
from pymanopt.manifolds import PSDFixedRank
from pymanopt.solvers import TrustRegions


def _bootstrap_problem(A, k):
    m, n = A.shape
    assert m == n, "matrix must be square"
    assert np.allclose(np.sum(A - A.T), 0), "matrix must be symmetric"
    manifold = PSDFixedRank(n, k)
    solver = TrustRegions(maxiter=500, minstepsize=1e-6)
    return manifold, solver


def low_rank_matrix_approximation(A, k):
    manifold, solver = _bootstrap_problem(A, k)

    @pymanopt.function.Callable
    def cost(Y):
        return la.norm(Y.dot(Y.T) - A, "fro") ** 2

    @pymanopt.function.Callable
    def egrad(Y):
        return 4 * (Y.dot(Y.T) - A).dot(Y)

    @pymanopt.function.Callable
    def ehess(Y, U):
        return 4 * ((Y.dot(U.T) + U.dot(Y.T)).dot(Y) + (Y.dot(Y.T) - A).dot(U))

    problem = pymanopt.Problem(manifold, cost, egrad=egrad, ehess=ehess)
    return solver.solve(problem)


def low_rank_matrix_approximation_theano(A, k):
    manifold, solver = _bootstrap_problem(A, k)

    Y = T.matrix()

    @pymanopt.function.Theano(Y)
    def cost(Y):
        return T.sum((T.dot(Y, Y.T) - A) ** 2)

    problem = pymanopt.Problem(manifold, cost)
    return solver.solve(problem)


if __name__ == "__main__":
    # Generate random problem data.
    n = 1000
    k = 5
    Y = rnd.randn(n, k)
    A = Y.dot(Y.T)

    # Solve the problem with pymanopt.
    Yopt = low_rank_matrix_approximation(A, k)
    print('')
    Yopt_theano = low_rank_matrix_approximation_theano(A, k)

    # Print information about the solution.
    print('')
    print("rank of Y: %d" % la.matrix_rank(Y))
    print("rank of Yopt: %d" % la.matrix_rank(Yopt))
    print("rank of Yopt_theano: %d" % la.matrix_rank(Yopt_theano))
