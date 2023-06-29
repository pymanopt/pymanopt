import autograd.numpy as np
import jax.numpy as jnp
import tensorflow as tf
import torch

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import Elliptope
from pymanopt.optimizers import ConjugateGradient


SUPPORTED_BACKENDS = ("autograd", "jax", "pytorch", "tensorflow")


def create_cost(manifold, epsilon, backend):
    if backend == "autograd":

        @pymanopt.function.autograd(manifold)
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

    elif backend == "jax":

        @pymanopt.function.jax(manifold)
        def cost(X):
            Y = X @ X.T
            s = jnp.triu(Y, 1).max()
            expY = jnp.exp((Y - s) / epsilon)
            expY -= jnp.diag(jnp.diag(expY))
            u = jnp.triu(expY, 1).sum()
            return s + epsilon * jnp.log(u)

    elif backend == "pytorch":

        @pymanopt.function.pytorch(manifold)
        def cost(X):
            Y = X @ torch.transpose(X, 1, 0)
            s = torch.triu(Y, 1).max()
            expY = torch.exp((Y - s) / epsilon)
            expY = expY - torch.diag(torch.diag(expY))
            u = torch.triu(expY, 1).sum()
            return s + epsilon * torch.log(u)

    elif backend == "tensorflow":

        @pymanopt.function.tensorflow(manifold)
        def cost(X):
            Y = X @ tf.transpose(X)
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

    optimizer = ConjugateGradient(
        min_gradient_norm=1e-8,
        max_iterations=1e5,
        verbosity=2 * int(not quiet),
    )
    Yopt = optimizer.run(problem).point

    if quiet:
        return

    Xopt = Yopt @ Yopt.T
    maxdot = np.triu(Xopt, 1).max()
    print("Maximum angle between any two points:", maxdot)


if __name__ == "__main__":
    runner = ExampleRunner(run, "Packing on the sphere", SUPPORTED_BACKENDS)
    runner.run()
