import os

import autograd.numpy as np
import tensorflow as tf
import theano.tensor as T
import torch
from examples._tools import ExampleRunner
from numpy import linalg as la, random as rnd

import pymanopt
from pymanopt.manifolds import FixedRankEmbedded
from pymanopt.solvers import ConjugateGradient


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


SUPPORTED_BACKENDS = (
    "Autograd", "Callable", "PyTorch", "TensorFlow", "Theano"
)


def create_cost_egrad(backend, A, rank):
    m, n = A.shape
    egrad = None

    if backend == "Autograd":
        @pymanopt.function.Autograd
        def cost(u, s, vt):
            X = u @ np.diag(s) @ vt
            return np.linalg.norm(X - A) ** 2
    elif backend == "Callable":
        @pymanopt.function.Callable
        def cost(u, s, vt):
            X = u @ np.diag(s) @ vt
            return la.norm(X - A) ** 2

        @pymanopt.function.Callable
        def egrad(u, s, vt):
            X = u @ np.diag(s) @ vt
            S = np.diag(s)
            gu = 2 * (X - A) @ (S @ vt).T
            gs = 2 * np.diag(u.T @ (X - A) @ vt.T)
            gvt = 2 * (u @ S).T @ (X - A)
            return gu, gs, gvt
    elif backend == "PyTorch":
        A_ = torch.from_numpy(A)

        @pymanopt.function.PyTorch
        def cost(u, s, vt):
            X = torch.matmul(u, torch.matmul(torch.diag(s), vt))
            return torch.norm(X - A_) ** 2
    elif backend == "TensorFlow":
        u = tf.Variable(tf.zeros((m, rank), dtype=np.float64), name="u")
        s = tf.Variable(tf.zeros(rank, dtype=np.float64), name="s")
        vt = tf.Variable(tf.zeros((rank, n), dtype=np.float64), name="vt")

        @pymanopt.function.TensorFlow(u, s, vt)
        def cost(u, s, vt):
            X = tf.matmul(u, tf.matmul(tf.linalg.diag(s), vt))
            return tf.norm(X - A) ** 2
    elif backend == "Theano":
        u = T.matrix()
        s = T.vector()
        vt = T.matrix()

        @pymanopt.function.Theano(u, s, vt)
        def cost(u, s, vt):
            X = T.dot(T.dot(u, T.diag(s)), vt)
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

    solver = ConjugateGradient()
    left_singular_vectors, singular_values, right_singular_vectors = \
        solver.solve(problem)
    low_rank_approximation = (left_singular_vectors @
                              np.diag(singular_values) @
                              right_singular_vectors)

    if not quiet:
        u, s, vt = la.svd(matrix, full_matrices=False)
        indices = np.argsort(s)[-rank:]
        low_rank_solution = (u[:, indices] @
                             np.diag(s[indices]) @
                             vt[indices, :])
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
