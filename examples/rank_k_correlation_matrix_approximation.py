import autograd.numpy as np
import jax.numpy as jnp
import tensorflow as tf
import torch

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import Oblique
from pymanopt.optimizers import TrustRegions


SUPPORTED_BACKENDS = ("autograd", "jax", "numpy", "pytorch", "tensorflow")


def create_cost_and_derivates(manifold, matrix, backend):
    euclidean_gradient = euclidean_hessian = None

    if backend == "autograd":

        @pymanopt.function.autograd(manifold)
        def cost(X):
            return 0.25 * np.linalg.norm(X.T @ X - matrix) ** 2

    elif backend == "jax":

        @pymanopt.function.jax(manifold)
        def cost(X):
            return 0.25 * jnp.linalg.norm(X.T @ X - matrix) ** 2

    elif backend == "numpy":

        @pymanopt.function.numpy(manifold)
        def cost(X):
            return 0.25 * np.linalg.norm(X.T @ X - matrix) ** 2

        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(X):
            return 0.5 * X @ (X.T @ X - matrix)

        @pymanopt.function.numpy(manifold)
        def euclidean_hessian(X, H):
            return X @ (H.T @ X + X.T @ H) + H @ (X.T @ X - matrix)

    elif backend == "pytorch":
        matrix_ = torch.from_numpy(matrix)

        @pymanopt.function.pytorch(manifold)
        def cost(X):
            return (
                0.25 * torch.norm(torch.transpose(X, 1, 0) @ X - matrix_) ** 2
            )

    elif backend == "tensorflow":

        @pymanopt.function.tensorflow(manifold)
        def cost(X):
            return 0.25 * tf.norm(tf.transpose(X) @ X - matrix) ** 2

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, euclidean_gradient, euclidean_hessian


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    num_rows = 10
    rank = 3
    matrix = np.random.normal(size=(num_rows, num_rows))
    matrix = 0.5 * (matrix + matrix.T)

    # Solve the problem with pymanopt.
    manifold = Oblique(rank, num_rows)
    cost, euclidean_gradient, euclidean_hessian = create_cost_and_derivates(
        manifold, matrix, backend
    )
    problem = pymanopt.Problem(
        manifold,
        cost,
        euclidean_gradient=euclidean_gradient,
        euclidean_hessian=euclidean_hessian,
    )

    optimizer = TrustRegions(verbosity=2 * int(not quiet))
    X = optimizer.run(problem).point

    if quiet:
        return

    C = X.T @ X
    print("Diagonal elements:", np.diag(C))
    print("Eigenvalues:", np.sort(np.linalg.eig(C)[0].real)[::-1])


if __name__ == "__main__":
    runner = ExampleRunner(
        run, "Nearest low-rank correlation matrix", SUPPORTED_BACKENDS
    )
    runner.run()
