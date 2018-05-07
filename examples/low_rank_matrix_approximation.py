import autograd.numpy as np

from pymanopt import Problem
from pymanopt.tools import decorators
from pymanopt.manifolds import FixedRankEmbedded
from pymanopt.solvers import ConjugateGradient

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
@decorators.autograd
def cost(usv):
    delta = .5
    u = usv[0]
    s = usv[1]
    vt = usv[2]
    X = np.dot(np.dot(u, np.diag(s)), vt)
    return np.sum(np.sqrt((X - A)**2 + delta**2) - delta)


# define the Pymanopt problem
problem = Problem(manifold=manifold, cost=cost)
# (c) Instantiation of a Pymanopt solver
solver = ConjugateGradient()

# let Pymanopt do the rest
X = solver.solve(problem)
