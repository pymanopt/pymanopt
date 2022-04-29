import autograd.numpy as np
import tensorflow as tf
import torch

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import Grassmann
from pymanopt.optimizers import TrustRegions


SUPPORTED_BACKENDS = ("autograd", "numpy", "pytorch", "tensorflow")


def create_cost_egrad_ehess(manifold, matrix, backend):
    egrad = ehess = None

    if backend == "autograd":

        @pymanopt.function.autograd(manifold)
        def cost(X):
            return -np.trace(X.T @ matrix @ X)

    elif backend == "numpy":

        @pymanopt.function.numpy(manifold)
        def cost(X):
            return -np.trace(X.T @ matrix @ X)

        @pymanopt.function.numpy(manifold)
        def egrad(X):
            return -(matrix + matrix.T) @ X

        @pymanopt.function.numpy(manifold)
        def ehess(X, H):
            return -(matrix + matrix.T) @ H

    elif backend == "pytorch":
        matrix_ = torch.from_numpy(matrix)

        @pymanopt.function.pytorch(manifold)
        def cost(X):
            return -torch.tensordot(X, torch.matmul(matrix_, X))

    elif backend == "tensorflow":

        @pymanopt.function.tensorflow(manifold)
        def cost(X):
            return -tf.tensordot(X, tf.matmul(matrix, X), axes=2)

        @pymanopt.function.tensorflow(manifold)
        def egrad(X):
            return -tf.matmul(matrix + matrix.T, X)

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, egrad, ehess


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    """Dominant invariant subspace example.

    This example generates a random 128 x 128 symmetric matrix, and finds the
    dominant invariant 3-dimensional subspace for this matrix.
    That is, it finds the subspace spanned by the three eigenvectors with the
    largest eigenvalues.
    """
    num_rows = 128
    subspace_dimension = 3
    matrix = np.random.normal(size=(num_rows, num_rows))
    matrix = 0.5 * (matrix + matrix.T)

    manifold = Grassmann(num_rows, subspace_dimension)
    cost, egrad, ehess = create_cost_egrad_ehess(manifold, matrix, backend)
    problem = pymanopt.Problem(manifold, cost=cost, egrad=egrad, ehess=ehess)

    optimizer = TrustRegions(verbosity=2 * int(not quiet))
    estimated_spanning_set = optimizer.run(
        problem, Delta_bar=8 * np.sqrt(subspace_dimension)
    )

    if quiet:
        return

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    column_indices = np.argsort(eigenvalues)[-subspace_dimension:]
    spanning_set = eigenvectors[:, column_indices]
    print(
        "Geodesic distance between true and estimated dominant subspace:",
        manifold.dist(spanning_set, estimated_spanning_set),
    )


if __name__ == "__main__":
    runner = ExampleRunner(
        run, "Dominant invariant subspace", SUPPORTED_BACKENDS
    )
    runner.run()
