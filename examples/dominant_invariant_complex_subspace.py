import autograd.numpy as np
import jax.numpy as jnp
import tensorflow as tf
import torch

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import ComplexGrassmann
from pymanopt.optimizers import TrustRegions


SUPPORTED_BACKENDS = ("autograd", "jax", "numpy", "pytorch", "tensorflow")


def create_cost_and_derivates(manifold, matrix, backend):
    euclidean_gradient = euclidean_hessian = None

    if backend == "autograd":

        @pymanopt.function.autograd(manifold)
        def cost(X):
            return -np.real(np.trace(np.conj(X.T) @ matrix @ X))

    elif backend == "jax":

        @pymanopt.function.jax(manifold)
        def cost(X):
            return -jnp.real(jnp.trace(jnp.conj(X.T) @ matrix @ X))

    elif backend == "numpy":

        @pymanopt.function.numpy(manifold)
        def cost(X):
            return -np.trace(X.T.conj() @ matrix @ X).real

        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(X):
            return -2 * matrix @ X

        @pymanopt.function.numpy(manifold)
        def euclidean_hessian(X, H):
            return -2 * matrix @ H

    elif backend == "pytorch":
        matrix_ = torch.from_numpy(matrix)

        @pymanopt.function.pytorch(manifold)
        def cost(X):
            return -torch.tensordot(X.conj(), matrix_ @ X).real

    elif backend == "tensorflow":

        @pymanopt.function.tensorflow(manifold)
        def cost(X):
            return -tf.math.real(
                tf.tensordot(tf.math.conj(X), matrix @ X, axes=2)
            )

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, euclidean_gradient, euclidean_hessian


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    num_rows = 128
    subspace_dimension = 3
    matrix = np.random.normal(
        size=(num_rows, num_rows)
    ) + 1j * np.random.normal(size=(num_rows, num_rows))
    matrix = 0.5 * (matrix + matrix.T.conj())

    manifold = ComplexGrassmann(num_rows, subspace_dimension)
    cost, euclidean_gradient, euclidean_hessian = create_cost_and_derivates(
        manifold, matrix, backend
    )
    problem = pymanopt.Problem(
        manifold,
        cost,
        euclidean_gradient=euclidean_gradient,
        euclidean_hessian=euclidean_hessian,
    )

    optimizer = TrustRegions(verbosity=2 * int(not quiet))
    estimated_spanning_set = optimizer.run(
        problem, Delta_bar=8 * np.sqrt(subspace_dimension)
    ).point

    if quiet:
        return

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    column_indices = np.argsort(eigenvalues)[-subspace_dimension:]
    spanning_set = eigenvectors[:, column_indices]
    print(
        "Geodesic distance between true and estimated dominant complex "
        "subspace:",
        manifold.dist(spanning_set, estimated_spanning_set),
    )


if __name__ == "__main__":
    runner = ExampleRunner(
        run, "Dominant invariant complex subspace", SUPPORTED_BACKENDS
    )
    runner.run()
