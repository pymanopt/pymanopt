import os

import autograd.numpy as np
import tensorflow as tf
import theano.tensor as T
import torch
from examples._tools import ExampleRunner
from numpy import linalg as la, random as rnd

import pymanopt
from pymanopt.manifolds import Oblique
from pymanopt.solvers import ConjugateGradient


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


SUPPORTED_BACKENDS = (
    "Autograd", "Callable", "PyTorch", "TensorFlow", "Theano"
)


def create_cost_egrad(backend, A):
    m, n = A.shape
    egrad = None

    if backend == "Autograd":
        @pymanopt.function.Autograd
        def cost(X):
            return 0.5 * np.sum((X - A) ** 2)
    elif backend == "Callable":
        @pymanopt.function.Callable
        def cost(X):
            return 0.5 * np.sum((X - A) ** 2)

        @pymanopt.function.Callable
        def egrad(X):
            return X - A
    elif backend == "PyTorch":
        A_ = torch.from_numpy(A)

        @pymanopt.function.PyTorch
        def cost(X):
            return 0.5 * torch.sum((X - A_) ** 2)
    elif backend == "TensorFlow":
        X = tf.Variable(tf.zeros((m, n), dtype=np.float64), name="X")

        @pymanopt.function.TensorFlow(X)
        def cost(X):
            return 0.5 * tf.reduce_sum((X - A) ** 2)
    elif backend == "Theano":
        X = T.matrix()

        @pymanopt.function.Theano(X)
        def cost(X):
            return 0.5 * T.sum((X - A) ** 2)
    else:
        raise ValueError("Unsupported backend '{:s}'".format(backend))

    return cost, egrad


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    m = 5
    n = 8
    matrix = rnd.randn(m, n)

    cost, egrad = create_cost_egrad(backend, matrix)
    manifold = Oblique(m, n)
    problem = pymanopt.Problem(manifold, cost=cost, egrad=egrad)
    if quiet:
        problem.verbosity = 0

    solver = ConjugateGradient()
    Xopt = solver.solve(problem)

    if quiet:
        return

    # Calculate the actual solution by normalizing the columns of A.
    X = matrix / la.norm(matrix, axis=0)[np.newaxis, :]

    # Print information about the solution.
    print("Solution found: %s" % np.allclose(X, Xopt, rtol=1e-3))
    print("Frobenius-error: %f" % la.norm(X - Xopt))


if __name__ == "__main__":
    runner = ExampleRunner(run, "Closest unit Frobenius norm approximation",
                           SUPPORTED_BACKENDS)
    runner.run()
