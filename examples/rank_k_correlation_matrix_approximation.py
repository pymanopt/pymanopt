import os

import autograd.numpy as np
import tensorflow as tf
import theano.tensor as T
import torch
from examples._tools import ExampleRunner
from numpy import linalg as la, random as rnd

import pymanopt
from pymanopt.manifolds import Oblique
from pymanopt.solvers import TrustRegions


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


SUPPORTED_BACKENDS = (
    "Autograd", "Callable", "PyTorch", "TensorFlow", "Theano"
)


def create_cost_egrad_ehess(backend, matrix, rank):
    num_rows = matrix.shape[0]
    egrad = ehess = None

    if backend == "Autograd":
        @pymanopt.function.Autograd
        def cost(X):
            return 0.25 * np.linalg.norm(X.T @ X - matrix) ** 2
    elif backend == "Callable":
        @pymanopt.function.Callable
        def cost(X):
            return 0.25 * np.linalg.norm(X.T @ X - matrix) ** 2

        @pymanopt.function.Callable
        def egrad(X):
            return 0.5 * X @ (X.T @ X - matrix)

        @pymanopt.function.Callable
        def ehess(X, H):
            return X @ (H.T @ X + X.T @ H) + H @ (X.T @ X - matrix)
    elif backend == "PyTorch":
        matrix_ = torch.from_numpy(matrix)

        @pymanopt.function.PyTorch
        def cost(X):
            return 0.25 * torch.norm(
                torch.matmul(torch.transpose(X, 1, 0), X) - matrix_) ** 2
    elif backend == "TensorFlow":
        X = tf.Variable(tf.zeros((rank, num_rows), dtype=np.float64), name="X")

        @pymanopt.function.TensorFlow(X)
        def cost(X):
            return 0.25 * tf.norm(tf.matmul(tf.transpose(X), X) - matrix) ** 2
    elif backend == "Theano":
        X = T.matrix()

        @pymanopt.function.Theano(X)
        def cost(X):
            return 0.25 * T.sum((T.dot(X.T, X) - matrix) ** 2)
    else:
        raise ValueError("Unsupported backend '{:s}'".format(backend))

    return cost, egrad, ehess


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    num_rows = 10
    rank = 3
    matrix = rnd.randn(num_rows, num_rows)
    matrix = 0.5 * (matrix + matrix.T)

    # Solve the problem with pymanopt.
    cost, egrad, ehess = create_cost_egrad_ehess(backend, matrix, rank)
    manifold = Oblique(rank, num_rows)
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
    runner = ExampleRunner(run, "Nearest low-rank correlation matrix",
                           SUPPORTED_BACKENDS)
    runner.run()
