import autograd.numpy as np
import tensorflow as tf
import torch
from examples._tools import ExampleRunner
from numpy import linalg as la, random as rnd

import pymanopt
from pymanopt.manifolds import Grassmann
from pymanopt.solvers import TrustRegions

SUPPORTED_BACKENDS = (
    "Autograd", "Callable", "PyTorch", "TensorFlow"
)


def create_cost_egrad_ehess(backend, A, p):
    egrad = ehess = None

    if backend == "Autograd":
        @pymanopt.function.Autograd
        def cost(X):
            return -np.trace(X.T @ A @ X)
    elif backend == "Callable":
        @pymanopt.function.Callable
        def cost(X):
            return -np.trace(X.T @ A @ X)

        @pymanopt.function.Callable
        def egrad(X):
            return -(A + A.T) @ X

        @pymanopt.function.Callable
        def ehess(X, H):
            return -(A + A.T) @ H
    elif backend == "PyTorch":
        A_ = torch.from_numpy(A)

        @pymanopt.function.PyTorch
        def cost(X):
            return -torch.tensordot(X, torch.matmul(A_, X))
    elif backend == "TensorFlow":
        @pymanopt.function.TensorFlow
        def cost(X):
            return -tf.tensordot(X, tf.matmul(A, X), axes=2)

        # Define the Euclidean gradient explicitly for the purpose of
        # demonstration. The Euclidean Hessian-vector product is automatically
        # calculated via TensorFlow's autodiff capabilities.
        @pymanopt.function.TensorFlow
        def egrad(X):
            return -tf.matmul(A + A.T, X)
    else:
        raise ValueError("Unsupported backend '{:s}'".format(backend))

    return cost, egrad, ehess


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    """This example generates a random 128 x 128 symmetric matrix and finds the
    dominant invariant 3 dimensional subspace for this matrix, i.e., it finds
    the subspace spanned by the three eigenvectors with the largest
    eigenvalues.
    """
    num_rows = 128
    subspace_dimension = 3
    matrix = rnd.randn(num_rows, num_rows)
    matrix = 0.5 * (matrix + matrix.T)

    cost, egrad, ehess = create_cost_egrad_ehess(
        backend, matrix, subspace_dimension)
    manifold = Grassmann(num_rows, subspace_dimension)
    problem = pymanopt.Problem(manifold, cost=cost, egrad=egrad, ehess=ehess)
    if quiet:
        problem.verbosity = 0

    solver = TrustRegions()
    estimated_spanning_set = solver.solve(
        problem, Delta_bar=8*np.sqrt(subspace_dimension))

    if quiet:
        return

    eigenvalues, eigenvectors = la.eig(matrix)
    column_indices = np.argsort(eigenvalues)[-subspace_dimension:]
    spanning_set = eigenvectors[:, column_indices]
    print("Geodesic distance between true and estimated dominant subspace:",
          manifold.dist(spanning_set, estimated_spanning_set))


if __name__ == "__main__":
    runner = ExampleRunner(run, "Dominant invariant subspace",
                           SUPPORTED_BACKENDS)
    runner.run()
