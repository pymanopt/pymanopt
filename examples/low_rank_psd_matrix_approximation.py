import autograd.numpy as np
import tensorflow as tf
import torch
from numpy import linalg as la
from numpy import random as rnd

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import PSDFixedRank
from pymanopt.solvers import TrustRegions


SUPPORTED_BACKENDS = ("Autograd", "Callable", "PyTorch", "TensorFlow")


def create_cost_egrad_ehess(manifold, matrix, backend):
    egrad = ehess = None

    if backend == "Autograd":

        @pymanopt.function.Autograd(manifold)
        def cost(Y):
            return np.linalg.norm(Y @ Y.T - matrix, "fro") ** 2

    elif backend == "Callable":

        @pymanopt.function.Callable(manifold)
        def cost(Y):
            return la.norm(Y @ Y.T - matrix, "fro") ** 2

        @pymanopt.function.Callable(manifold)
        def egrad(Y):
            return 4 * (Y @ Y.T - matrix) @ Y

        @pymanopt.function.Callable(manifold)
        def ehess(Y, U):
            return 4 * ((Y @ U.T + U @ Y.T) @ Y + (Y @ Y.T - matrix) @ U)

    elif backend == "PyTorch":
        matrix_ = torch.from_numpy(matrix)

        @pymanopt.function.PyTorch(manifold)
        def cost(Y):
            X = torch.matmul(Y, torch.transpose(Y, 1, 0))
            return torch.norm(X - matrix_) ** 2

    elif backend == "TensorFlow":

        @pymanopt.function.TensorFlow(manifold)
        def cost(Y):
            X = tf.matmul(Y, tf.transpose(Y))
            return tf.norm(X - matrix) ** 2

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, egrad, ehess


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    num_rows = 1000
    rank = 5
    low_rank_factor = rnd.randn(num_rows, rank)
    matrix = low_rank_factor @ low_rank_factor.T

    manifold = PSDFixedRank(num_rows, rank)
    cost, egrad, ehess = create_cost_egrad_ehess(manifold, matrix, backend)
    problem = pymanopt.Problem(manifold, cost=cost, egrad=egrad, ehess=ehess)
    if quiet:
        problem.verbosity = 0

    solver = TrustRegions(maxiter=500, minstepsize=1e-6)
    low_rank_factor_estimate = solver.solve(problem)

    if quiet:
        return

    print("Rank of target matrix:", la.matrix_rank(matrix))
    matrix_estimate = low_rank_factor_estimate @ low_rank_factor_estimate.T
    print(
        "Frobenius norm error of low-rank estimate:",
        la.norm(matrix - matrix_estimate),
    )


if __name__ == "__main__":
    runner = ExampleRunner(
        run, "Low-rank PSD matrix approximation", SUPPORTED_BACKENDS
    )
    runner.run()
