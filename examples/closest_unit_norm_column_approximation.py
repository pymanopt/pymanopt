import autograd.numpy as np
import jax.numpy as jnp
import tensorflow as tf
import torch

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import Oblique
from pymanopt.optimizers import ConjugateGradient


SUPPORTED_BACKENDS = ("autograd", "jax", "numpy", "pytorch", "tensorflow")


def create_cost_and_derivates(manifold, matrix, backend):
    euclidean_gradient = None

    if backend == "autograd":

        @pymanopt.function.autograd(manifold)
        def cost(X):
            return 0.5 * np.sum((X - matrix) ** 2)

    elif backend == "jax":

        @pymanopt.function.jax(manifold)
        def cost(X):
            return 0.5 * jnp.sum((X - matrix) ** 2)

    elif backend == "numpy":

        @pymanopt.function.numpy(manifold)
        def cost(X):
            return 0.5 * np.sum((X - matrix) ** 2)

        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(X):
            return X - matrix

    elif backend == "pytorch":
        matrix_ = torch.from_numpy(matrix)

        @pymanopt.function.pytorch(manifold)
        def cost(X):
            return 0.5 * torch.sum((X - matrix_) ** 2)

    elif backend == "tensorflow":

        @pymanopt.function.tensorflow(manifold)
        def cost(X):
            return 0.5 * tf.reduce_sum((X - matrix) ** 2)

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, euclidean_gradient


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    m = 5
    n = 8
    matrix = np.random.normal(size=(m, n))

    manifold = Oblique(m, n)
    cost, euclidean_gradient = create_cost_and_derivates(
        manifold, matrix, backend
    )
    problem = pymanopt.Problem(
        manifold, cost, euclidean_gradient=euclidean_gradient
    )

    optimizer = ConjugateGradient(
        verbosity=2 * int(not quiet), beta_rule="FletcherReeves"
    )
    Xopt = optimizer.run(problem).point

    if quiet:
        return

    # Calculate the actual solution by normalizing the columns of matrix.
    X = matrix / np.linalg.norm(matrix, axis=0)[np.newaxis, :]

    # Print information about the solution.
    print("Solution found:", np.allclose(X, Xopt, rtol=1e-3))
    print("Frobenius-error:", np.linalg.norm(X - Xopt))


if __name__ == "__main__":
    runner = ExampleRunner(
        run, "Closest unit Frobenius norm approximation", SUPPORTED_BACKENDS
    )
    runner.run()
