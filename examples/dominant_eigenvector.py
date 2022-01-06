import autograd.numpy as np
import tensorflow as tf
import torch
from numpy import linalg as la
from numpy import random as rnd

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import Sphere
from pymanopt.solvers import SteepestDescent


SUPPORTED_BACKENDS = ("Autograd", "Callable", "PyTorch", "TensorFlow")


def create_cost_egrad(manifold, matrix, backend):
    egrad = None

    if backend == "Autograd":

        @pymanopt.function.Autograd(manifold)
        def cost(x):
            return -np.inner(x, matrix @ x)

    elif backend == "Callable":

        @pymanopt.function.Callable(manifold)
        def cost(x):
            return -np.inner(x, matrix @ x)

        @pymanopt.function.Callable(manifold)
        def egrad(x):
            return -2 * matrix @ x

    elif backend == "PyTorch":
        matrix_ = torch.from_numpy(matrix)

        @pymanopt.function.PyTorch(manifold)
        def cost(x):
            return -torch.matmul(x, torch.matmul(matrix_, x))

    elif backend == "TensorFlow":

        @pymanopt.function.TensorFlow(manifold)
        def cost(x):
            return -tf.tensordot(x, tf.tensordot(matrix, x, axes=1), axes=1)

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, egrad


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    n = 128
    matrix = rnd.randn(n, n)
    matrix = 0.5 * (matrix + matrix.T)

    manifold = Sphere(n)
    cost, egrad = create_cost_egrad(manifold, matrix, backend)
    problem = pymanopt.Problem(manifold, cost=cost, egrad=egrad)
    if quiet:
        problem.verbosity = 0

    solver = SteepestDescent()
    estimated_dominant_eigenvector = solver.solve(problem)

    if quiet:
        return

    # Calculate the actual solution by a conventional eigenvalue decomposition.
    eigenvalues, eigenvectors = la.eig(matrix)
    dominant_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

    # Make sure both vectors have the same direction. Both are valid
    # eigenvectors, but for comparison we need to get rid of the sign
    # ambiguity.
    if np.sign(dominant_eigenvector[0]) != np.sign(
        estimated_dominant_eigenvector[0]
    ):
        estimated_dominant_eigenvector = -estimated_dominant_eigenvector

    # Print information about the solution.
    print("l2-norm of x:", la.norm(dominant_eigenvector))
    print("l2-norm of xopt:", la.norm(estimated_dominant_eigenvector))
    print(
        "Solution found:",
        np.allclose(
            dominant_eigenvector, estimated_dominant_eigenvector, atol=1e-6
        ),
    )
    error_norm = la.norm(dominant_eigenvector - estimated_dominant_eigenvector)
    print("l2-error:", error_norm)


if __name__ == "__main__":
    runner = ExampleRunner(
        run, "Dominant eigenvector of a PSD matrix", SUPPORTED_BACKENDS
    )
    runner.run()
