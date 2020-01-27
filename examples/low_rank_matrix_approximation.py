import autograd.numpy as np
import numpy.linalg as la
from theano import tensor as T

import pymanopt
from pymanopt.manifolds import FixedRankEmbedded
from pymanopt.solvers import ConjugateGradient


def main():
    # Let A be a (5 x 4) matrix to be approximated
    A = np.random.randn(5, 4)
    k = 2

    # (a) Instantiation of a manifold
    # points on the manifold are parameterized via their singular value
    # decomposition (u, s, vt) where
    # u is a 5 x 2 matrix with orthonormal columns,
    # s is a vector of length 2,
    # vt is a 2 x 4 matrix with orthonormal rows,
    # so that u*diag(s)*vt is a 5 x 4 matrix of rank 2.
    manifold = FixedRankEmbedded(A.shape[0], A.shape[1], k)

    # (b) Definition of a cost function (here using autograd.numpy)
    #       Note that the cost must be defined in terms of u, s and vt, where
    #       X = u * diag(s) * vt.
    @pymanopt.function.Autograd(("u", "s", "vt"))
    def cost(u, s, vt):
        delta = 0.5
        X = np.dot(np.dot(u, np.diag(s)), vt)
        return np.sum(np.sqrt((X - A) ** 2 + delta ** 2) - delta)

    # define the Pymanopt problem
    problem = pymanopt.Problem(manifold, cost)
    # (c) Instantiation of a Pymanopt solver
    solver = ConjugateGradient()

    # let Pymanopt do the rest
    print("Solving with autograd:")
    print()
    u, s, vt = solver.solve(problem)
    X_autograd = u @ np.diag(s) @ vt

    U = T.matrix()
    S = T.vector()
    Vt = T.matrix()

    @pymanopt.function.Theano((U, S, Vt))
    def cost(U, S, Vt):
        delta = 0.5
        X = T.dot(T.dot(U, T.diag(S)), Vt)
        return T.sum(T.sqrt((X - A) ** 2 + delta ** 2) - delta)

    problem = pymanopt.Problem(manifold, cost)

    print("Solving with theano:")
    print()
    u, s, vt = solver.solve(problem)
    X_theano = u @ np.diag(s) @ vt

    print("autograd solution:")
    print(X_autograd)
    print()
    print("theano solution:")
    print(X_theano)
    print()
    print("norm error between autograd and theano solution",
          la.norm(X_autograd - X_theano))


if __name__ == "__main__":
    main()
