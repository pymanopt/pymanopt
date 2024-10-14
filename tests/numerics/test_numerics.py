import jax.numpy as jnp
import numpy as np
import pytest
import scipy
import tensorflow as tf
import torch

from pymanopt.numerics import (
    JaxNumericsBackend,
    NumpyNumericsBackend,
    PytorchNumericsBackend,
    TensorflowNumericsBackend,
)
from pymanopt.numerics.core import NumericsBackend


backend_np64 = NumpyNumericsBackend(dtype=np.float64)
backend_np32 = NumpyNumericsBackend(dtype=np.float32)
backend_pt64 = PytorchNumericsBackend(dtype=torch.float64)
backend_pt32 = PytorchNumericsBackend(dtype=torch.float32)
backend_jnp64 = JaxNumericsBackend(dtype=jnp.float64)
backend_jnp32 = JaxNumericsBackend(dtype=jnp.float32)
backend_tf64 = TensorflowNumericsBackend(dtype=tf.float64)
backend_tf32 = TensorflowNumericsBackend(dtype=tf.float32)
all_backends = [
    backend_np64,
    backend_np32,
    backend_pt64,
    backend_pt32,
    backend_jnp64,
    backend_jnp32,
    # backend_tf64,
    # backend_tf32,
]


@pytest.mark.parametrize(
    "input, expected_output",
    [
        ([-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]),
        ([1, 0, 1], [1, 0, 1]),
        ([-127, -4], [127, 4]),
    ],
)
@pytest.mark.parametrize("backend", all_backends)
def test_abs(input, expected_output, backend):
    backend.assert_allclose(
        backend.abs(backend.array(input)), backend.array(expected_output)
    )


@pytest.mark.parametrize(
    "input, expected_output",
    [
        ([True, False, True], False),
        ([True, True, True], True),
        ([], True),
    ],
)
@pytest.mark.parametrize("backend", all_backends)
def test_all(input, expected_output, backend):
    assert backend.all(backend.array(input)) == expected_output


@pytest.mark.parametrize(
    "input, expected_output",
    [
        ([False, False, False], False),
        ([True, False, True], True),
        ([True, True, False], True),
        ([], False),
    ],
)
@pytest.mark.parametrize("backend", all_backends)
def test_any(input, expected_output, backend):
    assert backend.any(backend.array(input)) == expected_output


@pytest.mark.parametrize(
    "input, expected_output",
    [
        (2, [[1.0, 0.0], [0.0, 1.0]]),
        (3, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ],
)
@pytest.mark.parametrize("backend", all_backends)
def test_eye(input, expected_output, backend):
    output = backend.eye(input)
    expected_output = backend.array(expected_output)
    assert output.dtype == expected_output.dtype
    backend.assert_allclose(output, expected_output)


@pytest.mark.parametrize("backend", [backend_np64, backend_np32])
def test_multilogm_singlemat(backend: NumericsBackend):
    m = 40
    a = np.diag(np.random.uniform(size=m))
    q, _ = np.linalg.qr(np.random.normal(size=(m, m)))
    # A is a positive definite matrix
    A = q @ a @ q.T
    backend.assert_allclose(
        backend.linalg_logm(A, positive_definite=True), scipy.linalg.logm(A)
    )


@pytest.mark.parametrize("backend", [backend_np64, backend_np32])
def test_multilogm(backend: NumericsBackend):
    k = 10
    m = 40
    A = np.zeros((k, m, m))
    L = np.zeros((k, m, m))
    for i in range(k):
        a = np.diag(np.random.uniform(size=m))
        q, _ = np.linalg.qr(np.random.normal(size=(m, m)))
        A[i] = q @ a @ q.T
        L[i] = scipy.linalg.logm(A[i])
    backend.assert_allclose(backend.linalg_logm(A, positive_definite=True), L)


@pytest.mark.parametrize("backend", [backend_np64, backend_np32])
def test_multilogm_complex_positive_definite(backend: NumericsBackend):
    k = 10
    m = 40
    shape = (k, m, m)
    A = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
    A = A @ backend.conjugate_transpose(A)
    # Compare fast path for positive definite matrices vs. general slow
    # one.
    backend.assert_allclose(
        backend.linalg_logm(A, positive_definite=True),
        backend.linalg_logm(A, positive_definite=False),
    )
