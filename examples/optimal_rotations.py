import autograd.numpy as np
import jax.numpy as jnp
import tensorflow as tf
import torch

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import SpecialOrthogonalGroup
from pymanopt.optimizers import SteepestDescent


SUPPORTED_BACKENDS = ("autograd", "jax", "numpy", "pytorch", "tensorflow")


def create_cost_and_derivates(manifold, ABt, backend):
    euclidean_gradient = None

    if backend == "autograd":

        @pymanopt.function.autograd(manifold)
        def cost(X):
            return -np.tensordot(X, ABt, axes=X.ndim)

    elif backend == "jax":

        @pymanopt.function.jax(manifold)
        def cost(X):
            return -jnp.tensordot(X, ABt, axes=X.ndim)

    elif backend == "numpy":

        @pymanopt.function.numpy(manifold)
        def cost(X):
            return -np.tensordot(X, ABt, axes=X.ndim)

        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(X):
            return -ABt

    elif backend == "pytorch":
        ABt_ = torch.from_numpy(ABt)

        @pymanopt.function.pytorch(manifold)
        def cost(X):
            return -torch.tensordot(X, ABt_, dims=X.dim())

    elif backend == "tensorflow":

        @pymanopt.function.tensorflow(manifold)
        def cost(X):
            return -tf.tensordot(X, ABt, axes=ABt.ndim)

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, euclidean_gradient


def compute_optimal_solution(ABt):
    n = ABt[0].shape[0]
    U, S, Vt = np.linalg.svd(ABt)
    UVt = U @ Vt
    if abs(1.0 - np.linalg.det(UVt)) < 1e-10:
        return UVt
    # UVt is in O(n) but not SO(n), which is easily corrected.
    J = np.append(np.ones(n - 1), -1)
    return (U * J) @ Vt


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    n = 3
    m = 10
    k = 10

    A = np.random.normal(size=(k, n, m))
    B = np.random.normal(size=(k, n, m))
    ABt = np.array([Ak @ Bk.T for Ak, Bk in zip(A, B)])

    manifold = SpecialOrthogonalGroup(n, k=k)
    cost, euclidean_gradient = create_cost_and_derivates(
        manifold, ABt, backend
    )
    problem = pymanopt.Problem(
        manifold, cost, euclidean_gradient=euclidean_gradient
    )

    optimizer = SteepestDescent(verbosity=2 * int(not quiet))
    X = optimizer.run(problem).point

    if not quiet:
        Xopt = np.array([compute_optimal_solution(ABtk) for ABtk in ABt])
        print("Frobenius norm error:", np.linalg.norm(Xopt - X))


if __name__ == "__main__":
    runner = ExampleRunner(
        run, "Optimal rotations example", SUPPORTED_BACKENDS
    )
    runner.run()
