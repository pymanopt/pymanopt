import os

import autograd.numpy as np
import tensorflow as tf
import theano.tensor as T
import torch
from examples._tools import ExampleRunner
from numpy import linalg as la, random as rnd

import pymanopt
from pymanopt.manifolds import FixedRankEmbedded
from pymanopt.solvers import ConjugateGradients


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


SUPPORTED_BACKENDS = (
    "Autograd", "Callable", "PyTorch", "TensorFlow", "Theano"
)


def create_cost_egrad(backend, A, rank):
    m, n = A.shape
    egrad = None

    if backend == "Autograd":
        @pymanopt.function.Autograd
        def cost(u, s, v):
            X = u @ s @ v.T
            return np.linalg.norm(X - A) ** 2
    elif backend == "Callable":
        @pymanopt.function.Callable
        def cost(u, s, v):
            X = u @ s @ v.T
            return la.norm(X - A) ** 2

        @pymanopt.function.Callable
        def egrad(u, s, v):
            X = u @ s @ v.T
            gu = 2 * (X - A) @ v @ s
            gs = 2 * u.T @ (X - A) @ v
            gv = 2 * (X - A).T @ u @ s
            return gu, gs, gv
    elif backend == "PyTorch":
        A_ = torch.from_numpy(A)

        @pymanopt.function.PyTorch
        def cost(u, s, v):
            X = u @ s @ v.t()
            return torch.norm(X - A_) ** 2
    elif backend == "TensorFlow":
        u = tf.Variable(tf.zeros((m, rank), dtype=np.float64), name="u")
        s = tf.Variable(tf.zeros((rank, rank), dtype=np.float64), name="s")
        v = tf.Variable(tf.zeros((n, rank), dtype=np.float64), name="v")

        @pymanopt.function.TensorFlow
        def cost(u, s, v):
            X = u @ s @ tf.transpose(v)
            return tf.norm(X - A) ** 2
    elif backend == "Theano":
        u = T.matrix()
        s = T.matrix()
        v = T.matrix()

        @pymanopt.function.Theano(u, s, v)
        def cost(u, s, v):
            X = T.dot(T.dot(u, s), v.T)
            return (X - A).norm(2) ** 2
    else:
        raise ValueError("Unsupported backend '{:s}'".format(backend))

    return cost, egrad


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    m, n, rank = 5, 4, 2
    matrix = rnd.randn(m, n)

    cost, egrad = create_cost_egrad(backend, matrix, rank)
    manifold = FixedRankEmbedded(m, n, rank)
    problem = pymanopt.Problem(manifold, cost=cost, egrad=egrad)
    if quiet:
        problem.verbosity = 0

    solver = ConjugateGradients()
    left_singular_vectors, singular_values, right_singular_vectors = \
        solver.solve(problem)
    low_rank_approximation = (left_singular_vectors @
                              singular_values @
                              right_singular_vectors.T)

    if not quiet:
        u, s, vt = la.svd(matrix, full_matrices=False)
        low_rank_solution = u[:, :rank] @ np.diag(s[:rank]) @ vt[:rank, :]
        print("Analytic low-rank solution:")
        print()
        print(low_rank_solution)
        print()
        print("Rank-{} approximation:".format(rank))
        print()
        print(low_rank_approximation)
        print()
        print("Frobenius norm error:",
              la.norm(low_rank_approximation - low_rank_solution))
        print()


if __name__ == "__main__":
    runner = ExampleRunner(run, "Low-rank matrix approximation",
                           SUPPORTED_BACKENDS)
    runner.run()
