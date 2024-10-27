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
    backend_tf64,
    backend_tf32,
]


@pytest.mark.parametrize(
    "inp, expected_output",
    [
        ([-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]),
        ([1, 0, 1], [1, 0, 1]),
        ([-127, -4], [127, 4]),
    ],
)
@pytest.mark.parametrize("backend", all_backends)
def test_abs(inp, expected_output, backend):
    backend.assert_allclose(
        backend.abs(backend.array(inp)), backend.array(expected_output)
    )


@pytest.mark.parametrize(
    "inp, expected_output",
    [
        ([True, False, True], False),
        ([True, True, True], True),
        ([], True),
    ],
)
@pytest.mark.parametrize("backend", all_backends)
def test_all(inp, expected_output, backend):
    assert backend.all(inp) == expected_output


@pytest.mark.parametrize(
    "inp, expected_output",
    [
        ([False, False, False], False),
        ([True, False, True], True),
        ([True, True, False], True),
        ([], False),
    ],
)
@pytest.mark.parametrize("backend", all_backends)
def test_any(inp, expected_output, backend):
    assert backend.any(inp) == expected_output


@pytest.mark.parametrize(
    "inp, expected_output",
    [
        (2, [[1.0, 0.0], [0.0, 1.0]]),
        (3, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ],
)
@pytest.mark.parametrize("backend", all_backends)
def test_eye(inp, expected_output, backend):
    output = backend.eye(inp)
    expected_output = backend.array(expected_output)
    assert output.dtype == expected_output.dtype
    backend.assert_allclose(output, expected_output)


@pytest.mark.parametrize(
    "backend", [backend_np64, backend_np32, backend_jnp64, backend_pt64]
)
def test_logm(backend: NumericsBackend):
    # test for a single matrix
    m = 10
    a = backend.random_uniform(m)
    q, _ = backend.linalg_qr(backend.random_normal(size=(m, m)))
    # A is a positive definite matrix
    A = (q * a) @ q.T
    logmA = backend.array(scipy.linalg.logm(np.asarray(A)))
    backend.assert_allclose(
        backend.linalg_logm(A, positive_definite=True), logmA
    )

    # test for several stacked matrices
    k = 4
    m = 10
    A = []
    L = []
    for _ in range(k):
        a = backend.random_uniform(size=m)
        q, _ = backend.linalg_qr(backend.random_normal(size=(m, m)))
        A.append((q * a) @ q.T)
        L.append(backend.array(scipy.linalg.logm(np.asarray(A[-1]))))
    A = backend.stack(A)
    L = backend.stack(L)
    backend.assert_allclose(backend.linalg_logm(A, positive_definite=True), L)


@pytest.mark.parametrize("backend", [backend_np64, backend_np32])
def test_multilogm_complex_positive_definite(backend: NumericsBackend):
    k = 4
    m = 10
    shape = (k, m, m)
    A = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
    A = A @ backend.conjugate_transpose(A)
    # Compare fast path for positive definite matrices vs. general slow
    # one.
    backend.assert_allclose(
        backend.linalg_logm(A, positive_definite=True),
        backend.linalg_logm(A, positive_definite=False),
    )


def test_conj():
    real_backend: NumericsBackend = backend_np64
    complex_backend: NumericsBackend = real_backend.to_complex_backend()
    x = real_backend.random_randn(2, 2)
    y = complex_backend.random_randn(2, 2)
    for arr in [x, y]:
        complex_backend.assert_allclose(
            real_backend.transpose(arr), complex_backend.transpose(arr)
        )
        complex_backend.assert_allclose(
            real_backend.conjugate_transpose(arr),
            complex_backend.conjugate_transpose(arr),
        )
        print(real_backend.transpose(arr).dtype)
        print(real_backend.conjugate_transpose(arr).dtype)
        print(complex_backend.transpose(arr).dtype)
        print(complex_backend.conjugate_transpose(arr).dtype)
