import os

import autograd.numpy as np
import tensorflow as tf
import theano.tensor as T
import torch
from examples._tools import ExampleRunner
from numpy import linalg as la, random as rnd

import pymanopt
from pymanopt.manifolds import PSDFixedRank
from pymanopt.solvers import TrustRegions


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


SUPPORTED_BACKENDS = (
    "Autograd", "Callable", "PyTorch", "TensorFlow", "Theano"
)


def create_cost_egrad_ehess(backend, A, rank):
    num_rows = A.shape[0]
    egrad = ehess = None

    if backend == "Autograd":
        @pymanopt.function.Autograd
        def cost(Y):
            return np.linalg.norm(Y @ Y.T - A, "fro") ** 2
    elif backend == "Callable":
        @pymanopt.function.Callable
        def cost(Y):
            return la.norm(Y @ Y.T - A, "fro") ** 2

        @pymanopt.function.Callable
        def egrad(Y):
            return 4 * (Y @ Y.T - A) @ Y

        @pymanopt.function.Callable
        def ehess(Y, U):
            return 4 * ((Y @ U.T + U @ Y.T) @ Y + (Y @ Y.T - A) @ U)
    elif backend == "PyTorch":
        A_ = torch.from_numpy(A)

        @pymanopt.function.PyTorch
        def cost(Y):
            X = torch.matmul(Y, torch.transpose(Y, 1, 0))
            return torch.norm(X - A_) ** 2
    elif backend == "TensorFlow":
        Y = tf.Variable(tf.zeros((num_rows, rank), dtype=np.float64), name="Y")

        @pymanopt.function.TensorFlow(Y)
        def cost(Y):
            X = tf.matmul(Y, tf.transpose(Y))
            return tf.norm(X - A) ** 2
    elif backend == "Theano":
        Y = T.matrix()

        @pymanopt.function.Theano(Y)
        def cost(Y):
            return T.sum((T.dot(Y, Y.T) - A) ** 2)
    else:
        raise ValueError("Unsupported backend '{:s}'".format(backend))

    return cost, egrad, ehess


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    num_rows = 1000
    rank = 5
    low_rank_factor = rnd.randn(num_rows, rank)
    matrix = low_rank_factor @ low_rank_factor.T

    cost, egrad, ehess = create_cost_egrad_ehess(backend, matrix, rank)
    manifold = PSDFixedRank(num_rows, rank)
    problem = pymanopt.Problem(manifold, cost=cost, egrad=egrad, ehess=ehess)
    if quiet:
        problem.verbosity = 0

    solver = TrustRegions(maxiter=500, minstepsize=1e-6)
    low_rank_factor_estimate = solver.solve(problem)

    if quiet:
        return

    print("Rank of target matrix:", la.matrix_rank(matrix))
    matrix_estimate = low_rank_factor_estimate @ low_rank_factor_estimate.T
    print("Frobenius norm error of low-rank estimate:",
          la.norm(matrix - matrix_estimate))


if __name__ == "__main__":
    runner = ExampleRunner(run, "Low-rank PSD matrix approximation",
                           SUPPORTED_BACKENDS)
    runner.run()
