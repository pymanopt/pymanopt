import autograd.numpy as np
import tensorflow as tf
import torch
from numpy import linalg as la
from numpy import random as rnd

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import Oblique
from pymanopt.solvers import TrustRegions


SUPPORTED_BACKENDS = ("Autograd", "Callable", "PyTorch", "TensorFlow")


def create_cost_egrad_ehess(manifold, matrix, backend):
    egrad = ehess = None

    if backend == "Autograd":

        @pymanopt.function.Autograd(manifold)
        def cost(X):
            return 0.25 * np.linalg.norm(X.T @ X - matrix) ** 2

    elif backend == "Callable":

        @pymanopt.function.Callable(manifold)
        def cost(X):
            return 0.25 * np.linalg.norm(X.T @ X - matrix) ** 2

        @pymanopt.function.Callable(manifold)
        def egrad(X):
            return 0.5 * X @ (X.T @ X - matrix)

        @pymanopt.function.Callable(manifold)
        def ehess(X, H):
            return X @ (H.T @ X + X.T @ H) + H @ (X.T @ X - matrix)

    elif backend == "PyTorch":
        matrix_ = torch.from_numpy(matrix)

        @pymanopt.function.PyTorch(manifold)
        def cost(X):
            return (
                0.25
                * torch.norm(
                    torch.matmul(torch.transpose(X, 1, 0), X) - matrix_
                )
                ** 2
            )

    elif backend == "TensorFlow":

        @pymanopt.function.TensorFlow(manifold)
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
    if quiet:
        problem.verbosity = 0

    solver = TrustRegions()
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
