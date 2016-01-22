import numpy as np
import numpy.linalg as la
import numpy.random as rnd
import theano.tensor as T

from pymanopt import Problem
from pymanopt.manifolds import SymFixedRankYY
from pymanopt.solvers import TrustRegions


def _bootstrap_problem(A, k):
    m, n = A.shape
    assert m == n, "matrix must be square"
    assert np.allclose(np.sum(A - A.T), 0), "matrix must be symmetric"
    manifold = SymFixedRankYY(n, k)
    solver = TrustRegions(maxiter=500, minstepsize=1e-6)
    return manifold, solver

def low_rank_matrix_approximation(A, k):
    manifold, solver = _bootstrap_problem(A, k)

    def cost(Y):
        return la.norm(Y.dot(Y.T) - A, "fro") ** 2
    def egrad(Y):
        return 4 * (Y.dot(Y.T) - A).dot(Y)
    def ehess(Y, U):
        return 4 * ((Y.dot(U.T) + U.dot(Y.T)).dot(Y) + (Y.dot(Y.T) - A).dot(U))
    problem = Problem(man=manifold, cost=cost, egrad=egrad, ehess=ehess)
    return solver.solve(problem)

def low_rank_matrix_approximation_theano(A, k):
    manifold, solver = _bootstrap_problem(A, k)

    Y = T.matrix()
    cost = T.sum((T.dot(Y, Y.T) - A) ** 2)

    problem = Problem(man=manifold, theano_cost=cost, theano_arg=Y)
    return solver.solve(problem)

if __name__ == "__main__":
    # Generate random problem data.
    n = 1000
    k = 5
    Y = rnd.randn(n, k)
    A = Y.dot(Y.T)

    # Solve the problem with pymanopt.
    Yopt = low_rank_matrix_approximation(A, k)
    print
    Yopt_theano = low_rank_matrix_approximation_theano(A, k)

    # Print information about the solution.
    print
    print "rank of Y: %d" % la.matrix_rank(Y)
    print "rank of Yopt: %d" % la.matrix_rank(Yopt)
    print "rank of Yopt_theano: %d" % la.matrix_rank(Yopt_theano)
