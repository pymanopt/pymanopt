import autograd.numpy as np
import jax.numpy as jnp
import tensorflow as tf
import torch

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import FixedRankEmbedded
from pymanopt.optimizers import ConjugateGradient


SUPPORTED_BACKENDS = ("autograd", "jax", "numpy", "pytorch", "tensorflow")


def create_cost_and_derivates(manifold, matrix, backend):
    euclidean_gradient = None

    if backend == "autograd":

        @pymanopt.function.autograd(manifold)
        def cost(u, s, vt):
            X = u @ np.diag(s) @ vt
            return np.linalg.norm(X - matrix) ** 2

    elif backend == "jax":

        @pymanopt.function.jax(manifold)
        def cost(u, s, vt):
            X = u @ jnp.diag(s) @ vt
            return jnp.linalg.norm(X - matrix) ** 2

    elif backend == "numpy":

        @pymanopt.function.numpy(manifold)
        def cost(u, s, vt):
            X = u @ np.diag(s) @ vt
            return np.linalg.norm(X - matrix) ** 2

        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(u, s, vt):
            X = u @ np.diag(s) @ vt
            S = np.diag(s)
            gu = 2 * (X - matrix) @ (S @ vt).T
            gs = 2 * np.diag(u.T @ (X - matrix) @ vt.T)
            gvt = 2 * (u @ S).T @ (X - matrix)
            return gu, gs, gvt

    elif backend == "pytorch":
        matrix_ = torch.from_numpy(matrix)

        @pymanopt.function.pytorch(manifold)
        def cost(u, s, vt):
            X = u @ torch.diag(s) @ vt
            return torch.norm(X - matrix_) ** 2

    elif backend == "tensorflow":

        @pymanopt.function.tensorflow(manifold)
        def cost(u, s, vt):
            X = u @ tf.linalg.diag(s) @ vt
            return tf.norm(X - matrix) ** 2

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, euclidean_gradient


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    m, n, rank = 5, 4, 2
    matrix = np.random.normal(size=(m, n))

    manifold = FixedRankEmbedded(m, n, rank)
    cost, euclidean_gradient = create_cost_and_derivates(
        manifold, matrix, backend
    )
    problem = pymanopt.Problem(
        manifold, cost, euclidean_gradient=euclidean_gradient
    )

    optimizer = ConjugateGradient(
        verbosity=2 * int(not quiet), beta_rule="PolakRibiere"
    )
    (
        left_singular_vectors,
        singular_values,
        right_singular_vectors,
    ) = optimizer.run(problem).point
    low_rank_approximation = (
        left_singular_vectors
        @ np.diag(singular_values)
        @ right_singular_vectors
    )

    if not quiet:
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
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
            np.linalg.norm(low_rank_approximation - low_rank_solution),
        )
        print()


if __name__ == "__main__":
    runner = ExampleRunner(
        run, "Low-rank matrix approximation", SUPPORTED_BACKENDS
    )
    runner.run()
