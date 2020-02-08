import autograd.numpy as np
import tensorflow as tf
import theano.tensor as T
import torch

import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Sphere
from pymanopt.solvers import SteepestDescent
from pymanopt.tools.diagnostics import check_gradient


SUPPORTED_BACKENDS = (
    "Autograd", "Callable", "PyTorch", "TensorFlow", "Theano"
)


def create_cost_egrad(backend, A):
    m, n = A.shape
    egrad = None

    if backend == "Autograd":
        @pymanopt.function.Autograd
        def cost(x):
            return -np.inner(x,  A @ x)
    elif backend == "Callable":
        @pymanopt.function.Callable
        def cost(x):
            return -np.inner(x, A @ x)

        @pymanopt.function.Callable
        def egrad(x):
            return -2 * A @  x
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
    # Generate random problem data.
    n = 128
    A = np.random.randn(n, n)
    A = .5 * (A + A.T)
    cost, egrad = create_cost_egrad(backend, A)

    # Create the problem structure.
    manifold = Sphere(n)
    problem = Problem(manifold=manifold, cost=cost, egrad=egrad)

    # Numerically check gradient consistency (optional).
    check_gradient(problem)

    # Solve
    solver = SteepestDescent()
    _ = solver.solve(problem)


for backend in SUPPORTED_BACKENDS:
    print(backend)
    run(backend)
