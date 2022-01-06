import autograd.numpy as np
import tensorflow as tf
import torch

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import SpecialOrthogonalGroup
from pymanopt.solvers import SteepestDescent


SUPPORTED_BACKENDS = ("Autograd", "Callable", "PyTorch", "TensorFlow")


def create_cost_egrad(manifold, ABt, backend):
    egrad = None

    if backend == "Autograd":

        @pymanopt.function.Autograd(manifold)
        def cost(X):
            return -np.tensordot(X, ABt, axes=X.ndim)

    elif backend == "Callable":

        @pymanopt.function.Callable(manifold)
        def cost(X):
            return -np.tensordot(X, ABt, axes=X.ndim)

        @pymanopt.function.Callable(manifold)
        def egrad(X):
            return -ABt

    elif backend == "PyTorch":
        ABt_ = torch.from_numpy(ABt)

        @pymanopt.function.PyTorch(manifold)
        def cost(X):
            return -torch.tensordot(X, ABt_, dims=X.dim())

    elif backend == "TensorFlow":

        @pymanopt.function.TensorFlow(manifold)
        def cost(X):
            return -tf.tensordot(X, ABt, axes=ABt.ndim)

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, egrad


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

    A = np.random.randn(k, n, m)
    B = np.random.randn(k, n, m)
    ABt = np.array([Ak @ Bk.T for Ak, Bk in zip(A, B)])

    manifold = SpecialOrthogonalGroup(n, k)
    cost, egrad = create_cost_egrad(manifold, ABt, backend)
    problem = pymanopt.Problem(manifold, cost, egrad=egrad)
    if quiet:
        problem.verbosity = 0

    solver = SteepestDescent()
    X = solver.solve(problem)

    if not quiet:
        Xopt = np.array([compute_optimal_solution(ABtk) for ABtk in ABt])
        print("Frobenius norm error:", np.linalg.norm(Xopt - X))


if __name__ == "__main__":
    runner = ExampleRunner(
        run, "Optimal rotations example", SUPPORTED_BACKENDS
    )
    runner.run()
