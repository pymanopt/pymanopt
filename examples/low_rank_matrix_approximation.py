from pymanopt.manifolds import FixedRankEmbedded
import autograd.numpy as np
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient

# Let A be a (5 x 4) matrix to be approximated
A = np.array([[1., 0.], [0., 0.]])
k = 1

# (a) Instantiation of a manifold
# points on the manifold are parameterized as (U, S, V) where
# U is an orthonormal 5 x 2 matrix,
# S is a full rank diagonal 2 x 2 matrix,
# V is an orthonormal 4 x 2 matrix,
# such that U*S*V' is a 5 x 4 matrix of rank 2.
manifold = FixedRankEmbedded(A.shape[0], A.shape[1], k)


# (b) Definition of a cost function (here using autograd.numpy)
def cost(X):
    # delta = .5
    return np.linalg.norm(X - A) ** 2


# define the Pymanopt problem
problem = Problem(manifold=manifold, cost=cost)
# (c) Instantiation of a Pymanopt solver
solver = ConjugateGradient(minstepsize=0)

# let Pymanopt do the rest
X = solver.solve(problem)
