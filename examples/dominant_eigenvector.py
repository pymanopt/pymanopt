import autograd.numpy as np
import tensorflow as tf
import torch

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import Sphere
from pymanopt.optimizers import SteepestDescent


SUPPORTED_BACKENDS = ("autograd", "jax", "numpy", "pytorch", "tensorflow")


def create_cost_and_derivates(manifold, matrix, backend):
    euclidean_gradient = None

    if backend == "autograd":

        @pymanopt.function.autograd(manifold)
        def cost(x):
            return -x.T @ matrix @ x

    elif backend == "jax":

        @pymanopt.function.jax(manifold)
        def cost(x):
            return -x.T @ matrix @ x

    elif backend == "numpy":

        @pymanopt.function.numpy(manifold)
        def cost(x):
            return -x.T @ matrix @ x

        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(x):
            return -2 * matrix @ x

    elif backend == "pytorch":
        matrix_ = torch.from_numpy(matrix)

        @pymanopt.function.pytorch(manifold)
        def cost(x):
            return -x.t() @ matrix_ @ x

    elif backend == "tensorflow":

        @pymanopt.function.tensorflow(manifold)
        def cost(x):
            return -tf.tensordot(x, tf.tensordot(matrix, x, axes=1), axes=1)

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, euclidean_gradient


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    n = 128
    matrix = np.random.normal(size=(n, n))
    matrix = 0.5 * (matrix + matrix.T)

    manifold = Sphere(n)
    cost, euclidean_gradient = create_cost_and_derivates(
        manifold, matrix, backend
    )
    problem = pymanopt.Problem(
        manifold, cost, euclidean_gradient=euclidean_gradient
    )

    optimizer = SteepestDescent(verbosity=2 * int(not quiet))
    estimated_dominant_eigenvector = optimizer.run(problem).point

    if quiet:
        return

    # Calculate the actual solution by a conventional eigenvalue decomposition.
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    dominant_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

    # Make sure both vectors have the same direction. Both are valid
    # eigenvectors, but for comparison we need to get rid of the sign
    # ambiguity.
    if np.sign(dominant_eigenvector[0]) != np.sign(
        estimated_dominant_eigenvector[0]
    ):
        estimated_dominant_eigenvector = -estimated_dominant_eigenvector

    # Print information about the solution.
    print("l2-norm of x:", np.linalg.norm(dominant_eigenvector))
    print("l2-norm of xopt:", np.linalg.norm(estimated_dominant_eigenvector))
    print(
        "Solution found:",
        np.allclose(
            dominant_eigenvector, estimated_dominant_eigenvector, atol=1e-6
        ),
    )
    error_norm = np.linalg.norm(
        dominant_eigenvector - estimated_dominant_eigenvector
    )
    print("l2-error:", error_norm)


if __name__ == "__main__":
    runner = ExampleRunner(
        run, "Dominant eigenvector of a PSD matrix", SUPPORTED_BACKENDS
    )
    runner.run()
