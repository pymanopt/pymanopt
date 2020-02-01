import os

import autograd.numpy as np
import tensorflow as tf
import theano.tensor as T
import torch
from examples._tools import ExampleRunner

import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import TrustRegions


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


SUPPORTED_BACKENDS = (
    "Autograd", "Callable", "PyTorch", "TensorFlow", "Theano"
)


def create_cost_egrad_ehess(backend, samples, num_components):
    dimension = samples.shape[-1]
    egrad = ehess = None

    if backend == "Autograd":
        @pymanopt.function.Autograd
        def cost(w):
            return np.linalg.norm(samples - samples @ w @ w.T) ** 2
    elif backend == "Callable":
        @pymanopt.function.Callable
        def cost(w):
            return np.linalg.norm(samples - samples @ w @ w.T) ** 2

        @pymanopt.function.Callable
        def egrad(w):
            return -2 * (
                samples.T @ (samples - samples @ w @ w.T) +
                (samples - samples @ w @ w.T).T @ samples
            ) @ w

        @pymanopt.function.Callable
        def ehess(w, h):
            return -2 * (
                samples.T @ (samples - samples @ w @ h.T) @ w +
                samples.T @ (samples - samples @ h @ w.T) @ w +
                samples.T @ (samples - samples @ w @ w.T) @ h +
                (samples - samples @ w @ h.T).T @ samples @ w +
                (samples - samples @ h @ w.T).T @ samples @ w +
                (samples - samples @ w @ w.T).T @ samples @ h
            )
    elif backend == "PyTorch":
        samples_ = torch.from_numpy(samples)

        @pymanopt.function.PyTorch
        def cost(w):
            projector = torch.matmul(w, torch.transpose(w, 1, 0))
            return torch.norm(
                samples_ - torch.matmul(samples_, projector)) ** 2
    elif backend == "TensorFlow":
        w = tf.Variable(
            tf.zeros((dimension, num_components), dtype=np.float64), name="w")

        @pymanopt.function.TensorFlow(w)
        def cost(w):
            projector = tf.matmul(w, tf.transpose(w))
            return tf.norm(samples - tf.matmul(samples, projector)) ** 2
    elif backend == "Theano":
        w = T.matrix()

        @pymanopt.function.Theano(w)
        def cost(w):
            projector = T.dot(w, w.T)
            return (samples - T.dot(samples, projector)).norm(2) ** 2
    else:
        raise ValueError("Unsupported backend '{:s}'".format(backend))

    return cost, egrad, ehess


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    dimension = 3
    num_samples = 200
    num_components = 2
    samples = np.random.randn(num_samples, dimension) @ np.diag([3, 2, 1])
    samples -= samples.mean(axis=0)

    cost, egrad, ehess = create_cost_egrad_ehess(
        backend, samples, num_components)
    manifold = Stiefel(dimension, num_components)
    problem = pymanopt.Problem(manifold, cost, egrad=egrad, ehess=ehess)
    if quiet:
        problem.verbosity = 0

    solver = TrustRegions()
    # from pymanopt.solvers import ConjugateGradient
    # solver = ConjugateGradient()
    estimated_span_matrix = solver.solve(problem)

    if quiet:
        return

    estimated_projector = estimated_span_matrix @ estimated_span_matrix.T

    eigenvalues, eigenvectors = np.linalg.eig(samples.T @ samples)
    indices = np.argsort(eigenvalues)[::-1][:num_components]
    span_matrix = eigenvectors[:, indices]
    projector = span_matrix @ span_matrix.T

    print("Frobenius norm error between estimated and closed-form projection "
          "matrix:", np.linalg.norm(projector - estimated_projector))


if __name__ == "__main__":
    runner = ExampleRunner(run, "PCA", SUPPORTED_BACKENDS)
    runner.run()
