import os

import autograd.numpy as np
import tensorflow as tf
import theano.tensor as T
import torch
from examples._tools import ExampleRunner
from numpy import linalg as la, random as rnd

import pymanopt
from pymanopt.manifolds import Sphere
from pymanopt.solvers import SteepestDescent


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


SUPPORTED_BACKENDS = (
    "Autograd", "Callable", "PyTorch", "TensorFlow", "Theano"
)


def create_cost_egrad(backend, A):
    m, n = A.shape
    egrad = None

    if backend == "Autograd":
        @pymanopt.function.Autograd
        def cost(x):
            return -np.inner(x, A @ x)
    elif backend == "Callable":
        @pymanopt.function.Callable
        def cost(x):
            return -np.inner(x, A @ x)

        @pymanopt.function.Callable
        def egrad(x):
            return -2 * A @ x
    elif backend == "PyTorch":
        A_ = torch.from_numpy(A)

        @pymanopt.function.PyTorch
        def cost(x):
            return -torch.matmul(x, torch.matmul(A_, x))
    elif backend == "TensorFlow":
        x = tf.Variable(tf.zeros(n, dtype=np.float64), name="X")

        @pymanopt.function.TensorFlow(x)
        def cost(x):
            return -tf.tensordot(x, tf.tensordot(A, x, axes=1), axes=1)
    elif backend == "Theano":
        x = T.vector()

        @pymanopt.function.Theano(x)
        def cost(x):
            return -x.T.dot(T.dot(A, x))
    else:
        raise ValueError("Unsupported backend '{:s}'".format(backend))

    return cost, egrad


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    n = 128
    matrix = rnd.randn(n, n)
    matrix = 0.5 * (matrix + matrix.T)

    cost, egrad = create_cost_egrad(backend, matrix)
    manifold = Sphere(n)
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
    if (np.sign(dominant_eigenvector[0]) !=
            np.sign(estimated_dominant_eigenvector[0])):
        estimated_dominant_eigenvector = -estimated_dominant_eigenvector

    # Print information about the solution.
    print("l2-norm of x: %f" % la.norm(dominant_eigenvector))
    print("l2-norm of xopt: %f" % la.norm(estimated_dominant_eigenvector))
    print("Solution found: %s" % np.allclose(
        dominant_eigenvector, estimated_dominant_eigenvector, rtol=1e-3))
    error_norm = la.norm(
        dominant_eigenvector - estimated_dominant_eigenvector)
    print("l2-error: %f" % error_norm)


if __name__ == "__main__":
    runner = ExampleRunner(run, "Dominant eigenvector of a PSD matrix",
                           SUPPORTED_BACKENDS)
    runner.run()
