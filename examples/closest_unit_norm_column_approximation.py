import autograd.numpy as np
import tensorflow as tf
import torch
from numpy import linalg as la

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import Oblique
from pymanopt.optimizers import ConjugateGradient


SUPPORTED_BACKENDS = ("autograd", "numpy", "pytorch", "tensorflow")


def create_cost_egrad(manifold, matrix, backend):
    egrad = None

    if backend == "autograd":

        @pymanopt.function.autograd(manifold)
        def cost(X):
            return 0.5 * np.sum((X - matrix) ** 2)

    elif backend == "numpy":

        @pymanopt.function.numpy(manifold)
        def cost(X):
            return 0.5 * np.sum((X - matrix) ** 2)

        @pymanopt.function.numpy(manifold)
        def egrad(X):
            return X - matrix

    elif backend == "pytorch":
        matrix_ = torch.from_numpy(matrix)

        @pymanopt.function.pytorch(manifold)
        def cost(X):
            return 0.5 * torch.sum((X - matrix_) ** 2)

    elif backend == "tensorflow":

        @pymanopt.function.tensorflow(manifold)
        def cost(X):
            return 0.5 * tf.reduce_sum((X - matrix) ** 2)

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, egrad


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    m = 5
    n = 8
    matrix = np.random.randn(m, n)

    manifold = Oblique(m, n)
    cost, egrad = create_cost_egrad(manifold, matrix, backend)
    problem = pymanopt.Problem(manifold, cost=cost, egrad=egrad)

    optimizer = ConjugateGradient(verbosity=2 * int(not quiet))
    Xopt = optimizer.run(problem)

    if quiet:
        return

    # Calculate the actual solution by normalizing the columns of matrix.
    X = matrix / la.norm(matrix, axis=0)[np.newaxis, :]

    # Print information about the solution.
    print("Solution found: %s" % np.allclose(X, Xopt, rtol=1e-3))
    print("Frobenius-error: %f" % la.norm(X - Xopt))


if __name__ == "__main__":
    runner = ExampleRunner(
        run, "Closest unit Frobenius norm approximation", SUPPORTED_BACKENDS
    )
    runner.run()
