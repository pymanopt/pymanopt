from pymanopt.manifolds import FixedRankEmbedded
import autograd.numpy as np
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent

# Let A be a (5 x 4) matrix to be approximated
A = np.random.randn(5, 4)
k = 2

# (a) Instantiation of a manifold
# points on the manifold are parameterized as (U, S, V) where
# U is an orthonormal 5 x 2 matrix,
# S is a full rank diagonal 2 x 2 matrix,
# V is an orthonormal 4 x 2 matrix,
# such that U*S*V' is a 5 x 4 matrix of rank 2.
manifold = FixedRankEmbedded(A.shape[0], A.shape[1], k)


# (b) Definition of a cost function (here using autograd.numpy)
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
solver = SteepestDescent()

# let Pymanopt do the rest
X = solver.solve(problem)
