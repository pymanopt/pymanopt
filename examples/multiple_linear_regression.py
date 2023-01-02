import autograd.numpy as np
import jax.numpy as jnp
import tensorflow as tf
import torch

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import Euclidean
from pymanopt.optimizers import TrustRegions


SUPPORTED_BACKENDS = ("autograd", "jax", "numpy", "pytorch", "tensorflow")


def create_cost_and_derivates(manifold, samples, targets, backend):
    euclidean_gradient = euclidean_hessian = None

    if backend == "autograd":

        @pymanopt.function.autograd(manifold)
        def cost(weights):
            return np.linalg.norm(targets - samples @ weights) ** 2

    elif backend == "jax":

        @pymanopt.function.jax(manifold)
        def cost(weights):
            return jnp.linalg.norm(targets - samples @ weights) ** 2

    elif backend == "numpy":

        @pymanopt.function.numpy(manifold)
        def cost(weights):
            return np.linalg.norm(targets - samples @ weights) ** 2

        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(weights):
            return -2 * samples.T @ (targets - samples @ weights)

        @pymanopt.function.numpy(manifold)
        def euclidean_hessian(weights, vector):
            return 2 * samples.T @ samples @ vector

    elif backend == "pytorch":
        samples_ = torch.from_numpy(samples)
        targets_ = torch.from_numpy(targets)

        @pymanopt.function.pytorch(manifold)
        def cost(weights):
            return torch.norm(targets_ - samples_ @ weights) ** 2

    elif backend == "tensorflow":

        @pymanopt.function.tensorflow(manifold)
        def cost(weights):
            return (
                tf.norm(targets - tf.tensordot(samples, weights, axes=1)) ** 2
            )

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, euclidean_gradient, euclidean_hessian


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    num_samples, num_weights = 200, 3

    optimizer = TrustRegions(verbosity=0)
    manifold = Euclidean(3)

    for k in range(5):
        samples = np.random.normal(size=(num_samples, num_weights))
        targets = np.random.normal(size=num_samples)

        (
            cost,
            euclidean_gradient,
            euclidean_hessian,
        ) = create_cost_and_derivates(manifold, samples, targets, backend)
        problem = pymanopt.Problem(
            manifold,
            cost,
            euclidean_gradient=euclidean_gradient,
            euclidean_hessian=euclidean_hessian,
        )

        estimated_weights = optimizer.run(problem).point
        if not quiet:
            print(f"Run {k + 1}")
            print(
                "Weights found by pymanopt (top) / "
                "closed form solution (bottom)"
            )
            print(estimated_weights)
            print(np.linalg.pinv(samples) @ targets)
            print("")


if __name__ == "__main__":
    runner = ExampleRunner(
        run, "Multiple linear regression", SUPPORTED_BACKENDS
    )
    runner.run()
