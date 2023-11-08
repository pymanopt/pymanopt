import numpy as np
import pytest
from numpy import testing as np_testing
from scipy.linalg import expm, logm

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
