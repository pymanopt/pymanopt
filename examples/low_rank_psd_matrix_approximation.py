import autograd.numpy as np
import jax.numpy as jnp
import tensorflow as tf
import torch

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import PSDFixedRank
from pymanopt.optimizers import TrustRegions


SUPPORTED_BACKENDS = ("autograd", "jax", "numpy", "pytorch", "tensorflow")


def create_cost_and_derivates(manifold, matrix, backend):
    euclidean_gradient = euclidean_hessian = None

    if backend == "autograd":

        @pymanopt.function.autograd(manifold)
        def cost(Y):
            return np.linalg.norm(Y @ Y.T - matrix, "fro") ** 2

    elif backend == "jax":

        @pymanopt.function.jax(manifold)
        def cost(Y):
            return jnp.linalg.norm(Y @ Y.T - matrix, "fro") ** 2

    elif backend == "numpy":

        @pymanopt.function.numpy(manifold)
        def cost(Y):
            return np.linalg.norm(Y @ Y.T - matrix, "fro") ** 2

        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(Y):
            return 4 * (Y @ Y.T - matrix) @ Y

        @pymanopt.function.numpy(manifold)
        def euclidean_hessian(Y, U):
            return 4 * ((Y @ U.T + U @ Y.T) @ Y + (Y @ Y.T - matrix) @ U)

    elif backend == "pytorch":
        matrix_ = torch.from_numpy(matrix)

        @pymanopt.function.pytorch(manifold)
        def cost(Y):
            X = Y @ torch.transpose(Y, 1, 0)
            return torch.norm(X - matrix_) ** 2

    elif backend == "tensorflow":

        @pymanopt.function.tensorflow(manifold)
        def cost(Y):
            X = Y @ tf.transpose(Y)
            return tf.norm(X - matrix) ** 2

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, euclidean_gradient, euclidean_hessian


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    num_rows = 1000
    rank = 5
    low_rank_factor = np.random.normal(size=(num_rows, rank))
    matrix = low_rank_factor @ low_rank_factor.T

    manifold = PSDFixedRank(num_rows, rank)
    cost, euclidean_gradient, euclidean_hessian = create_cost_and_derivates(
        manifold, matrix, backend
    )
    problem = pymanopt.Problem(
        manifold,
        cost,
        euclidean_gradient=euclidean_gradient,
        euclidean_hessian=euclidean_hessian,
    )

    optimizer = TrustRegions(
        max_iterations=500, min_step_size=1e-6, verbosity=2 * int(not quiet)
    )
    low_rank_factor_estimate = optimizer.run(problem).point

    if quiet:
        return

    print("Rank of target matrix:", np.linalg.matrix_rank(matrix))
    matrix_estimate = low_rank_factor_estimate @ low_rank_factor_estimate.T
    print(
        "Frobenius norm error of low-rank estimate:",
        np.linalg.norm(matrix - matrix_estimate),
    )


if __name__ == "__main__":
    runner = ExampleRunner(
        run, "Low-rank PSD matrix approximation", SUPPORTED_BACKENDS
    )
    runner.run()
