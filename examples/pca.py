import autograd.numpy as np
import jax.numpy as jnp
import tensorflow as tf
import torch

import pymanopt
from examples._tools import ExampleRunner
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import TrustRegions


SUPPORTED_BACKENDS = ("autograd", "jax", "numpy", "pytorch", "tensorflow")


def create_cost_and_derivates(manifold, samples, backend):
    euclidean_gradient = euclidean_hessian = None

    if backend == "autograd":

        @pymanopt.function.autograd(manifold)
        def cost(w):
            return np.linalg.norm(samples - samples @ w @ w.T) ** 2

    elif backend == "jax":

        @pymanopt.function.jax(manifold)
        def cost(w):
            return jnp.linalg.norm(samples - samples @ w @ w.T) ** 2

    elif backend == "numpy":

        @pymanopt.function.numpy(manifold)
        def cost(w):
            return np.linalg.norm(samples - samples @ w @ w.T) ** 2

        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(w):
            return (
                -2
                * (
                    samples.T @ (samples - samples @ w @ w.T)
                    + (samples - samples @ w @ w.T).T @ samples
                )
                @ w
            )

        @pymanopt.function.numpy(manifold)
        def euclidean_hessian(w, h):
            return -2 * (
                samples.T @ (samples - samples @ w @ h.T) @ w
                + samples.T @ (samples - samples @ h @ w.T) @ w
                + samples.T @ (samples - samples @ w @ w.T) @ h
                + (samples - samples @ w @ h.T).T @ samples @ w
                + (samples - samples @ h @ w.T).T @ samples @ w
                + (samples - samples @ w @ w.T).T @ samples @ h
            )

    elif backend == "pytorch":
        samples_ = torch.from_numpy(samples)

        @pymanopt.function.pytorch(manifold)
        def cost(w):
            projector = w @ torch.transpose(w, 1, 0)
            return torch.norm(samples_ - samples_ @ projector) ** 2

    elif backend == "tensorflow":

        @pymanopt.function.tensorflow(manifold)
        def cost(w):
            projector = w @ tf.transpose(w)
            return tf.norm(samples - samples @ projector) ** 2

    else:
        raise ValueError(f"Unsupported backend '{backend}'")

    return cost, euclidean_gradient, euclidean_hessian


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    dimension = 3
    num_samples = 200
    num_components = 2
    samples = np.random.normal(size=(num_samples, dimension)) @ np.diag(
        [3, 2, 1]
    )
    samples -= samples.mean(axis=0)

    manifold = Stiefel(dimension, num_components)
    cost, euclidean_gradient, euclidean_hessian = create_cost_and_derivates(
        manifold, samples, backend
    )
    problem = pymanopt.Problem(
        manifold,
        cost,
        euclidean_gradient=euclidean_gradient,
        euclidean_hessian=euclidean_hessian,
    )

    optimizer = TrustRegions(verbosity=2 * int(not quiet))
    estimated_span_matrix = optimizer.run(problem).point

    if quiet:
        return

    estimated_projector = estimated_span_matrix @ estimated_span_matrix.T

    eigenvalues, eigenvectors = np.linalg.eig(samples.T @ samples)
    indices = np.argsort(eigenvalues)[::-1][:num_components]
    span_matrix = eigenvectors[:, indices]
    projector = span_matrix @ span_matrix.T

    print(
        "Frobenius norm error between estimated and closed-form projection "
        "matrix:",
        np.linalg.norm(projector - estimated_projector),
    )


if __name__ == "__main__":
    runner = ExampleRunner(run, "PCA", SUPPORTED_BACKENDS)
    runner.run()
