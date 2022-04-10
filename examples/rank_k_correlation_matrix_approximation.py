import autograd.numpy as np
import tensorflow as tf
import torch
from numpy import linalg as la
from numpy import random as rnd

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import Oblique
from pymanopt.solvers import TrustRegions


SUPPORTED_BACKENDS = ("autograd", "numpy", "pytorch", "tensorflow")


def create_cost_egrad_ehess(manifold, matrix, backend):
    egrad = ehess = None

    if backend == "autograd":

        @pymanopt.function.autograd(manifold)
        def cost(X):
            return 0.25 * np.linalg.norm(X.T @ X - matrix) ** 2

    elif backend == "numpy":

        @pymanopt.function.numpy(manifold)
        def cost(X):
            return 0.25 * np.linalg.norm(X.T @ X - matrix) ** 2

        @pymanopt.function.numpy(manifold)
        def egrad(X):
            return 0.5 * X @ (X.T @ X - matrix)

        @pymanopt.function.numpy(manifold)
        def ehess(X, H):
            return X @ (H.T @ X + X.T @ H) + H @ (X.T @ X - matrix)

    elif backend == "pytorch":
        matrix_ = torch.from_numpy(matrix)

        @pymanopt.function.pytorch(manifold)
        def cost(X):
            return (
                0.25
                * torch.norm(
                    torch.matmul(torch.transpose(X, 1, 0), X) - matrix_
                )
                ** 2
            )

    elif backend == "tensorflow":

        @pymanopt.function.tensorflow(manifold)
        def cost(X):
            return 0.25 * tf.norm(tf.matmul(tf.transpose(X), X) - matrix) ** 2

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, egrad, ehess


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    num_rows = 10
    rank = 3
    matrix = rnd.randn(num_rows, num_rows)
    matrix = 0.5 * (matrix + matrix.T)

    # Solve the problem with pymanopt.
    manifold = Oblique(rank, num_rows)
    cost, egrad, ehess = create_cost_egrad_ehess(manifold, matrix, backend)
    problem = pymanopt.Problem(manifold, cost, egrad=egrad, ehess=ehess)

    solver = TrustRegions(verbosity=2 * int(not quiet))
    X = solver.solve(problem)

    if quiet:
        return

    C = X.T @ X
    print("Diagonal elements:", np.diag(C))
    print("Eigenvalues:", np.sort(la.eig(C)[0].real)[::-1])


if __name__ == "__main__":
    runner = ExampleRunner(
        run, "Nearest low-rank correlation matrix", SUPPORTED_BACKENDS
    )
    runner.run()
