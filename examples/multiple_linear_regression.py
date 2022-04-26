import autograd.numpy as np
import tensorflow as tf
import torch
from numpy import linalg as la
from numpy import random as rnd

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import Euclidean
from pymanopt.optimizers import TrustRegions


SUPPORTED_BACKENDS = ("autograd", "numpy", "pytorch", "tensorflow")


def create_cost_egrad_ehess(manifold, samples, targets, backend):
    egrad = ehess = None

    if backend == "autograd":

        @pymanopt.function.autograd(manifold)
        def cost(weights):
            # Use autograd's linalg.norm wrapper.
            return np.linalg.norm(targets - samples @ weights) ** 2

    elif backend == "numpy":

        @pymanopt.function.numpy(manifold)
        def cost(weights):
            return la.norm(targets - samples @ weights) ** 2

        @pymanopt.function.numpy(manifold)
        def egrad(weights):
            return -2 * samples.T @ (targets - samples @ weights)

        @pymanopt.function.numpy(manifold)
        def ehess(weights, vector):
            return 2 * samples.T @ samples @ vector

    elif backend == "pytorch":
        samples_ = torch.from_numpy(samples)
        targets_ = torch.from_numpy(targets)

        @pymanopt.function.pytorch(manifold)
        def cost(weights):
            return torch.norm(targets_ - torch.matmul(samples_, weights)) ** 2

    elif backend == "tensorflow":

        @pymanopt.function.tensorflow(manifold)
        def cost(weights):
            return (
                tf.norm(targets - tf.tensordot(samples, weights, axes=1)) ** 2
            )

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, egrad, ehess


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    num_samples, num_weights = 200, 3

    optimizer = TrustRegions(verbosity=0)
    manifold = Euclidean(3)

    for k in range(5):
        samples = rnd.randn(num_samples, num_weights)
        targets = rnd.randn(num_samples)

        cost, egrad, ehess = create_cost_egrad_ehess(
            manifold, samples, targets, backend
        )
        problem = pymanopt.Problem(manifold, cost, egrad=egrad, ehess=ehess)

        estimated_weights = optimizer.run(problem)
        if not quiet:
            print(f"Run {k + 1}")
            print(
                "Weights found by pymanopt (top) / "
                "closed form solution (bottom)"
            )
            print(estimated_weights)
            print(la.pinv(samples) @ targets)
            print("")


if __name__ == "__main__":
    runner = ExampleRunner(
        run, "Multiple linear regression", SUPPORTED_BACKENDS
    )
    runner.run()
