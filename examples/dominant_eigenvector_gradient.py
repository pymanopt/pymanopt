import autograd.numpy as np

from pymanopt import Problem
from pymanopt.manifolds import Sphere
from pymanopt.solvers import TrustRegions
from pymanopt.tools.utils import checkgradient

# Generate random problem data.
n = 1000
A = np.random.randn(n)
A = .5 * (A + A.T)

# Create the problem structure.
manifold = Sphere(n)


def cost(x):
    # Define the problem cost function and its Euclidean gradient.
    return -x.T @ (A * x)


def egrad(x):
    # notice the 'e' in 'egrad' for Euclidean
    return -2 * A * x


problem = Problem(manifold=manifold, cost=cost, egrad=egrad)

# Numerically check gradient consistency (optional).
checkgradient(problem)

# Solve
solver = TrustRegions()
wopt = solver.solve(problem)
