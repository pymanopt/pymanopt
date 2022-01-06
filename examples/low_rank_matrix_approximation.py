import autograd.numpy as np
import tensorflow as tf
import torch
from numpy import linalg as la
from numpy import random as rnd

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import FixedRankEmbedded
from pymanopt.solvers import ConjugateGradient


SUPPORTED_BACKENDS = ("Autograd", "Callable", "PyTorch", "TensorFlow")


def create_cost_egrad(manifold, matrix, backend):
    egrad = None

    if backend == "Autograd":

        @pymanopt.function.Autograd(manifold)
        def cost(u, s, vt):
            X = u @ np.diag(s) @ vt
            return np.linalg.norm(X - matrix) ** 2

    elif backend == "Callable":

        @pymanopt.function.Callable(manifold)
        def cost(u, s, vt):
            X = u @ np.diag(s) @ vt
            return la.norm(X - matrix) ** 2

        @pymanopt.function.Callable(manifold)
        def egrad(u, s, vt):
            X = u @ np.diag(s) @ vt
            S = np.diag(s)
            gu = 2 * (X - matrix) @ (S @ vt).T
            gs = 2 * np.diag(u.T @ (X - matrix) @ vt.T)
            gvt = 2 * (u @ S).T @ (X - matrix)
            return gu, gs, gvt

    elif backend == "PyTorch":
        matrix_ = torch.from_numpy(matrix)

        @pymanopt.function.PyTorch(manifold)
        def cost(u, s, vt):
            X = torch.matmul(u, torch.matmul(torch.diag(s), vt))
            return torch.norm(X - matrix_) ** 2

    elif backend == "TensorFlow":

        @pymanopt.function.TensorFlow(manifold)
        def cost(u, s, vt):
            X = tf.matmul(u, tf.matmul(tf.linalg.diag(s), vt))
            return tf.norm(X - matrix) ** 2

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, egrad


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    m, n, rank = 5, 4, 2
    matrix = rnd.randn(m, n)

    manifold = FixedRankEmbedded(m, n, rank)
    cost, egrad = create_cost_egrad(manifold, matrix, backend)
    problem = pymanopt.Problem(manifold, cost=cost, egrad=egrad)
    if quiet:
        problem.verbosity = 0

    solver = ConjugateGradient()
    (
        left_singular_vectors,
        singular_values,
        right_singular_vectors,
    ) = solver.solve(problem)
    low_rank_approximation = (
        left_singular_vectors
        @ np.diag(singular_values)
        @ right_singular_vectors
    )

    if not quiet:
        u, s, vt = la.svd(matrix, full_matrices=False)
        indices = np.argsort(s)[-rank:]
        low_rank_solution = (
            u[:, indices] @ np.diag(s[indices]) @ vt[indices, :]
        )
        print("Analytic low-rank solution:")
        print()
        print(low_rank_solution)
        print()
        print(f"Rank-{rank} approximation:")
        print()
        print(low_rank_approximation)
        print()
        print(
            "Frobenius norm error:",
            la.norm(low_rank_approximation - low_rank_solution),
        )
        print()


if __name__ == "__main__":
    runner = ExampleRunner(
        run, "Low-rank matrix approximation", SUPPORTED_BACKENDS
    )
    runner.run()
