import autograd.numpy as np
import tensorflow as tf
import torch

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import Elliptope
from pymanopt.solvers import ConjugateGradient


SUPPORTED_BACKENDS = ("Autograd", "PyTorch", "TensorFlow")


def create_cost(manifold, epsilon, backend):
    if backend == "Autograd":

        @pymanopt.function.Autograd(manifold)
        def cost(X):
            Y = X @ X.T
            # Shift the exponentials by the maximum value to reduce numerical
            # trouble due to possible overflows.
            s = np.triu(Y, 1).max()
            expY = np.exp((Y - s) / epsilon)
            # Zero out the diagonal
            expY -= np.diag(np.diag(expY))
            u = np.triu(expY, 1).sum()
            return s + epsilon * np.log(u)

    elif backend == "PyTorch":

        @pymanopt.function.PyTorch(manifold)
        def cost(X):
            Y = torch.matmul(X, torch.transpose(X, 1, 0))
            s = torch.triu(Y, 1).max()
            expY = torch.exp((Y - s) / epsilon)
            expY = expY - torch.diag(torch.diag(expY))
            u = torch.triu(expY, 1).sum()
            return s + epsilon * torch.log(u)

    elif backend == "TensorFlow":

        @pymanopt.function.TensorFlow(manifold)
        def cost(X):
            Y = tf.matmul(X, tf.transpose(X))
            s = tf.reduce_max(tf.linalg.band_part(Y, 0, -1))
            expY = tf.exp((Y - s) / epsilon)
            expY = expY - tf.linalg.diag(tf.linalg.diag_part(expY))
            u = tf.reduce_sum(tf.linalg.band_part(Y, 0, -1))
            return s + epsilon * tf.math.log(u)

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    dimension = 3  # Dimension of the embedding space, i.e. R^k
    num_points = 24  # Points on the sphere
    # This value should be as close to 0 as affordable. If it is too close to
    # zero, optimization first becomes much slower, than simply doesn't work
    # anymore because of floating point overflow errors (NaN's and Inf's start
    # to appear). If it is too large, then log-sum-exp is a poor approximation
    # of the max function, and the spread will be less uniform. An okay value
    # seems to be 0.01 or 0.001 for example. Note that a better strategy than
    # using a small epsilon straightaway is to reduce epsilon bit by bit and to
    # warm-start subsequent optimization in that way. Trustregions will be more
    # appropriate for these fine tunings.
    epsilon = 0.005

    manifold = Elliptope(num_points, dimension)
    cost = create_cost(manifold, epsilon, backend)
    problem = pymanopt.Problem(manifold, cost)
    if quiet:
        problem.verbosity = 0

    solver = ConjugateGradient(mingradnorm=1e-8, maxiter=1e5)
    Yopt = solver.solve(problem)

    if quiet:
        return

    Xopt = Yopt @ Yopt.T
    maxdot = np.triu(Xopt, 1).max()
    print("Maximum angle between any two points:", maxdot)


if __name__ == "__main__":
    runner = ExampleRunner(run, "Packing on the sphere", SUPPORTED_BACKENDS)
    runner.run()
