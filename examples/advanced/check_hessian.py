import autograd.numpy as anp
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch

import pymanopt
from examples._tools import ExampleRunner
from pymanopt import Problem
from pymanopt.manifolds import Sphere
from pymanopt.tools.diagnostics import check_hessian


SUPPORTED_BACKENDS = ("autograd", "jax", "numpy", "pytorch", "tensorflow")


def create_cost_and_derivates(manifold, matrix, backend):
    euclidean_gradient = euclidean_hessian = None

    if backend == "autograd":

        @pymanopt.function.autograd(manifold)
        def cost(x):
            return -anp.inner(x, matrix @ x)

    elif backend == "jax":

        @pymanopt.function.jax(manifold)
        def cost(x):
            return -jnp.inner(x, matrix @ x)

    elif backend == "numpy":

        @pymanopt.function.numpy(manifold)
        def cost(x):
            return -np.inner(x, matrix @ x)

        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(x):
            return -2 * matrix @ x

        @pymanopt.function.numpy(manifold)
        def euclidean_hessian(x, d):
            return -2 * matrix @ d

    elif backend == "pytorch":
        matrix_ = torch.from_numpy(matrix)

        @pymanopt.function.pytorch(manifold)
        def cost(x):
            return -x @ matrix_ @ x

    elif backend == "tensorflow":

        @pymanopt.function.tensorflow(manifold)
        def cost(x):
            return -tf.tensordot(x, tf.tensordot(matrix, x, axes=1), axes=1)

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, euclidean_gradient, euclidean_hessian


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    n = 128
    manifold = Sphere(n)

    # Generate random problem data.
    matrix = np.random.normal(size=(n, n))
    matrix = 0.5 * (matrix + matrix.T)
    cost, euclidean_gradient, euclidean_hessian = create_cost_and_derivates(
        manifold, matrix, backend
    )

    # Create the problem structure.
    problem = Problem(
        manifold,
        cost,
        euclidean_gradient=euclidean_gradient,
        euclidean_hessian=euclidean_hessian,
    )

    # Numerically check gradient consistency (optional).
    check_hessian(problem)


if __name__ == "__main__":
    runner = ExampleRunner(
        run, "Check Hessian on sphere manifold", SUPPORTED_BACKENDS
    )
    runner.run()
