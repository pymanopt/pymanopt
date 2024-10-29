import numpy as np
import pytest
import scipy

from pymanopt.backends import Backend


@pytest.mark.parametrize(
    "inp, expected_output",
    [
        ([-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]),
        ([1, 0, 1], [1, 0, 1]),
        ([-127, -4], [127, 4]),
    ],
)
def test_abs(inp, expected_output, any_backend: Backend):
    bk = any_backend
    bk.assert_allclose(bk.abs(bk.array(inp)), bk.array(expected_output))


@pytest.mark.parametrize(
    "inp, expected_output",
    [
        ([True, False, True], False),
        ([True, True, True], True),
        ([], True),
    ],
)
def test_all(inp, expected_output, any_backend: Backend):
    assert any_backend.all(inp) == expected_output


@pytest.mark.parametrize(
    "inp, expected_output",
    [
        ([False, False, False], False),
        ([True, False, True], True),
        ([True, True, False], True),
        ([], False),
    ],
)
def test_any(inp, expected_output, any_backend: Backend):
    assert any_backend.any(inp) == expected_output


@pytest.mark.parametrize(
    "inp, expected_output",
    [
        (2, [[1.0, 0.0], [0.0, 1.0]]),
        (3, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ],
)
def test_eye(inp, expected_output, any_backend: Backend):
    bk = any_backend
    output = bk.eye(inp)
    expected_output = bk.array(expected_output)
    assert output.dtype == expected_output.dtype
    bk.assert_allclose(output, expected_output)


def test_logm(real_backend: Backend):
    bk = real_backend
    # test for a single matrix
    m = 10
    a = bk.reshape(bk.random_uniform(m), (-1, 1))
    q, _ = bk.linalg_qr(bk.random_normal(size=(m, m)))
    # A is a positive definite matrix
    A = q @ (a * bk.conjugate_transpose(q))
    logmA = bk.array(scipy.linalg.logm(np.asarray(A)))
    bk.assert_allclose(
        bk.linalg_logm(A, positive_definite=True), logmA, atol=1e-3
    )

    # test for several stacked matrices
    k = 4
    m = 10
    A = []
    L = []
    for _ in range(k):
        a = bk.reshape(bk.random_uniform(size=m), (-1, 1))
        q, _ = bk.linalg_qr(bk.random_normal(size=(m, m)))
        A.append(q @ (a * bk.conjugate_transpose(q)))
        L.append(bk.array(scipy.linalg.logm(np.asarray(A[-1]))))
    A = bk.stack(A)
    L = bk.stack(L)
    bk.assert_allclose(bk.linalg_logm(A, positive_definite=True), L, atol=1e-3)


@pytest.mark.skip
def test_multilogm_complex_positive_definite(complex_backend: Backend):
    bk = complex_backend
    k = 4
    m = 10
    shape = (k, m, m)
    A = bk.random_normal(size=shape)
    A = A @ bk.conjugate_transpose(A)
    # Compare fast path for positive definite matrices vs. general slow one.
    bk.assert_allclose(
        bk.linalg_logm(A, positive_definite=True),
        bk.linalg_logm(A, positive_definite=False),
    )


def test_conj(real_backend: Backend):
    complex_backend = real_backend.to_complex_backend()
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
