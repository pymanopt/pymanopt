import numpy as np
import pytest
import scipy.linalg

from pymanopt.tools.multi import (
    multiexpm,
    multieye,
    multihconj,
    multilogm,
    multiqr,
    multisym,
    multitransp,
)
import pymanopt.numerics as nx

from tests.numerics import _test_numerics_supported_backends


def parametrize_test_multitransp_singlemat(m, n):
    def wrapper(test_func):
        np.random.seed(0)
        A = np.random.normal(size=(m, n))
        params = [(A, A.T)]
        return pytest.mark.parametrize("A, expected_output", params)(test_func)

    return wrapper


@parametrize_test_multitransp_singlemat(m=40, n=50)
@_test_numerics_supported_backends()
def test_multitransp_singlemat(A, expected_output):
    output = multitransp(A)
    assert nx.allclose(output, expected_output)


def parametrize_test_multitransp(k, m, n):
    def wrapper(test_func):
        np.random.seed(0)
        A = np.random.normal(size=(k, m, n))

        C = np.zeros((k, n, m))
        for i in range(k):
            C[i] = A[i].T
        params = [(A, C)]
        return pytest.mark.parametrize("A, expected_output", params)(test_func)

    return wrapper


@parametrize_test_multitransp(k=5, m=40, n=50)
@_test_numerics_supported_backends()
def test_multitransp(A, expected_output):
    output = multitransp(A)
    assert nx.allclose(output, expected_output)


def parametrize_test_multisym(k, m):
    def wrapper(test_func):
        np.random.seed(0)
        A = np.random.normal(size=(k, m, m))

        C = np.zeros((k, m, m))
        for i in range(k):
            C[i] = (A[i] + A[i].T) / 2
        params = [(A, C)]
        return pytest.mark.parametrize("A, expected_output", params)(test_func)

    return wrapper


@parametrize_test_multisym(k=5, m=40)
@_test_numerics_supported_backends()
def test_multisym(A, expected_output):
    output = multisym(A)
    assert nx.allclose(output, expected_output)


def parametrize_test_multieye(k, m):
    def wrapper(test_func):
        C = np.zeros((k, m, m))
        for i in range(k):
            C[i] = np.eye(m)
        params = [(k, m, C)]
        return pytest.mark.parametrize("k, m, expected_output", params)(test_func)

    return wrapper


@parametrize_test_multieye(k=5, m=40)
@_test_numerics_supported_backends()
def test_multieye(k, m, expected_output):
    output = multieye(k, m)
    assert nx.allclose(output, expected_output)


def parametrize_test_multilogm_singlemat(m):
    def wrapper(test_func):
        np.random.seed(0)
        a = np.diag(np.random.uniform(size=m))
        q, _ = np.linalg.qr(np.random.normal(size=(m, m)))
        # A is a positive definite matrix
        A = q @ a @ q.T
        L = scipy.linalg.logm(A)
        params = [(A, L)]
        return pytest.mark.parametrize("A, expected_output", params)(test_func)

    return wrapper


@parametrize_test_multilogm_singlemat(m=40)
@_test_numerics_supported_backends()
def test_multilogm_singlemat(A, expected_output):
    output = multilogm(A, positive_definite=True)
    assert nx.allclose(output, expected_output)


def parametrize_test_multilogm(k, m):
    def wrapper(test_func):
        np.random.seed(0)
        A = np.zeros((k, m, m))
        L = np.zeros((k, m, m))
        for i in range(k):
            a = np.diag(np.random.uniform(size=m))
            q, _ = np.linalg.qr(np.random.normal(size=(m, m)))
            A[i] = q @ a @ q.T
            L[i] = scipy.linalg.logm(A[i])
        params = [(A, L)]
        return pytest.mark.parametrize("A, expected_output", params)(test_func)

    return wrapper


@parametrize_test_multilogm(k=5, m=40)
@_test_numerics_supported_backends()
def test_multilogm(A, expected_output):
    output = multilogm(A, positive_definite=True)
    assert nx.allclose(output, expected_output)


def parametrize_test_multilogm_complex_positive_definite(k, m):
    def wrapper(test_func):
        np.random.seed(0)
        shape = (k, m, m)
        A = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
        A = A @ multihconj(A)
        L = np.zeros(shape, dtype=np.complex128)
        for i in range(k):
            L[i] = scipy.linalg.logm(A[i])
        params = [(A, L)]
        return pytest.mark.parametrize("A, expected_output", params)(test_func)

    return wrapper


@parametrize_test_multilogm_complex_positive_definite(k=5, m=40)
@_test_numerics_supported_backends()
def test_multilogm_complex_positive_definite(A, expected_output):
    output = multilogm(A, positive_definite=False)
    assert nx.allclose(output, expected_output)

    output = multilogm(A, positive_definite=True)
    assert nx.allclose(output, expected_output)


def parametrize_test_multiexpm_singlemat(m):
    def wrapper(test_func):
        np.random.seed(0)
        A = np.random.normal(size=(m, m))
        A = A + A.T
        L = scipy.linalg.expm(A)
        params = [(A, L)]
        return pytest.mark.parametrize("A, expected_output", params)(test_func)

    return wrapper


@parametrize_test_multiexpm_singlemat(m=40)
@_test_numerics_supported_backends()
def test_multiexpm_singlemat(A, expected_output):
    output = multiexpm(A, symmetric=True)
    assert nx.allclose(output, expected_output)


def parametrize_test_multiexpm(k, m):
    def wrapper(test_func):
        np.random.seed(0)
        A = multisym(np.random.normal(size=(k, m, m)))
        E = np.zeros((k, m, m))
        for i in range(k):
            E[i] = scipy.linalg.expm(A[i])
        params = [(A, E)]
        return pytest.mark.parametrize("A, expected_output", params)(test_func)

    return wrapper


@parametrize_test_multiexpm(k=5, m=40)
@_test_numerics_supported_backends()
def test_multiexpm(A, expected_output):
    output = multiexpm(A, symmetric=True)
    assert nx.allclose(output, expected_output)


def parametrize_test_multiexpm_conjugate_symmetric(k, m):
    def wrapper(test_func):
        np.random.seed(0)
        shape = (k, m, m)
        A = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
        A = A + multihconj(A) / 2
        E = np.zeros(shape, dtype=np.complex128)
        for i in range(k):
            E[i] = scipy.linalg.expm(A[i])
        params = [(A, E)]
        return pytest.mark.parametrize("A, expected_output", params)(test_func)

    return wrapper


@parametrize_test_multiexpm(k=5, m=40)
@_test_numerics_supported_backends()
def test_multiexpm_conjugate_symmetric(A, expected_output):
    output = multiexpm(A, symmetric=False)
    assert nx.allclose(output, expected_output)

    output = multiexpm(A, symmetric=True)
    assert nx.allclose(output, expected_output)


def parametrize_test_multiqr_singlemat(m, n):
    def wrapper(test_func):
        np.random.seed(0)
        A = np.random.normal(size=(m, n))
        params = [(A)]
        return pytest.mark.parametrize("A", params)(test_func)

    return wrapper


@parametrize_test_multiqr_singlemat(m=40, n=50)
@_test_numerics_supported_backends()
def test_multiqr_singlemat(A):
    Q, R = multiqr(A)
    assert nx.allclose(Q @ R, A)


def parametrize_test_multiqr_singlemat_complex(m, n):
    def wrapper(test_func):
        np.random.seed(0)
        A = np.random.normal(size=(m, n)) + 1j * np.random.normal(size=(m, n))
        params = [(A)]
        return pytest.mark.parametrize("A", params)(test_func)

    return wrapper


@parametrize_test_multiqr_singlemat_complex(m=40, n=50)
@_test_numerics_supported_backends()
def test_multiqr_singlemat_complex(A):
    Q, R = multiqr(A)
    assert nx.allclose(Q @ R, A)


def parametrize_test_multiqr(k, m, n):
    def wrapper(test_func):
        np.random.seed(0)
        A = np.random.normal(size=(k, m, n)) + 1j * np.random.normal(size=(k, m, n))
        params = [(A)]
        return pytest.mark.parametrize("A", params)(test_func)

    return wrapper


@parametrize_test_multiqr(k=5, m=40, n=50)
@_test_numerics_supported_backends()
def test_multiqr(A):
    Q, R = multiqr(A)
    assert nx.allclose(Q @ R, A)
