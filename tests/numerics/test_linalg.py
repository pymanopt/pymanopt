import numpy as np
import pytest
import scipy

import pymanopt.numerics as nx
from tests.numerics import _test_numerics_supported_backends


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([[1]]), np.linalg.cholesky(np.array([[1]]))),
        (np.array([[2, 1], [1, 2]]), np.linalg.cholesky(np.array([[2, 1], [1, 2]]))),
    ],
)
@_test_numerics_supported_backends()
def test_cholesky(argument, expected_output):
    output = nx.linalg.cholesky(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([[1]]), np.linalg.det(np.array([[1]]))),
        (np.array([[2, 1], [1, 2]]), np.linalg.det(np.array([[2, 1], [1, 2]]))),
    ],
)
@_test_numerics_supported_backends()
def test_det(argument, expected_output):
    output = nx.linalg.det(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([[2, 1], [1, 2]]), np.linalg.eigh(np.array([[2, 1], [1, 2]]))),
    ],
)
@_test_numerics_supported_backends()
def test_eigh(argument, expected_output):
    eigenvalues, eigenvectors = nx.linalg.eigh(argument)
    assert nx.allclose(eigenvalues, expected_output[0])
    assert nx.allclose(eigenvectors, expected_output[1])


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([[1]]), scipy.linalg.expm(np.array([[1]]))),
        (
            np.array([[1, 1.3], [1.3, 1]]),
            scipy.linalg.expm(np.array([[1, 1.3], [1.3, 1]]))
        ),
    ],
)
@_test_numerics_supported_backends()
def test_expm(argument, expected_output):
    output = nx.linalg.expm(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([[1]]), np.linalg.inv(np.array([[1]]))),
        (
            np.array([[2, 1], [1, 2]]),
            np.linalg.inv(np.array([[2, 1], [1, 2]]))
        ),
    ],
)
@_test_numerics_supported_backends()
def test_inv(argument, expected_output):
    output = nx.linalg.inv(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([[1]]), scipy.linalg.logm(np.array([[1]]))),
        (
            np.array([[2, 1], [1, 2]]),
            scipy.linalg.logm(np.array([[2, 1], [1, 2]]))
        ),
    ],
)
@_test_numerics_supported_backends()
def test_logm(argument, expected_output):
    output = nx.linalg.logm(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([[1]]), np.linalg.matrix_rank(np.array([[1]]))),
        (
            np.array([[2, 1], [4, 2]]),
            np.linalg.matrix_rank(np.array([[2, 1], [4, 2]]))
        ),
        (
            np.array([[2, 1], [4, 1]]),
            np.linalg.matrix_rank(np.array([[2, 1], [4, 1]]))
        ),
    ]
)
@_test_numerics_supported_backends()
def test_matrix_rank(argument, expected_output):
    output = nx.linalg.matrix_rank(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument_a, argument_b, expected_output",
    [
        (1, None, np.linalg.norm(1)),
        ([2], None, np.linalg.norm([2])),
        (np.array([1, 4.2]), None, np.linalg.norm(np.array([1, 4.2]))),
        (
            np.array([[1, 2], [3, 4.2]]), 0,
            np.linalg.norm(np.array([[1, 2], [3, 4.2]]), axis=0)
        ),
    ]
)
@_test_numerics_supported_backends()
def test_norm(argument_a, argument_b, expected_output):
    output = nx.linalg.norm(argument_a, axis=argument_b)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([[1]]), np.linalg.qr(np.array([[1]]))),
        (
            np.array([[2, 1], [1, 2]]),
            np.linalg.qr(np.array([[2, 1], [1, 2]]))
        ),
    ],
)
@_test_numerics_supported_backends()
def test_qr(argument, expected_output):
    output = nx.linalg.qr(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument_a, argument_b, expected_output",
    [
        (
            np.array([[1]]), np.array([[1]]),
            np.linalg.solve(np.array([[1]]), np.array([[1]]))
        ),
        (
            np.array([[2, 1], [1, 2]]), -np.array([[4.2, 0.3], [0.3, 4.2]]),
            np.linalg.solve(
                np.array([[2, 1], [1, 2]]), -np.array([[4.2, 0.3], [0.3, 4.2]]))
        ),
    ],
)
@_test_numerics_supported_backends()
def test_solve(argument_a, argument_b, expected_output):
    output = nx.linalg.solve(argument_a, argument_b)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument_a, argument_b, expected_output",
    [
        (
            np.array([[1]]), np.array([[3]]),
            scipy.linalg.solve_continuous_lyapunov(
                np.array([[1]]), np.array([[3]]))
        ),
        (
            np.array([[2, 1], [1, 2]]), np.array([[1.2, 4], [2, 1]]),
            scipy.linalg.solve_continuous_lyapunov(
                np.array([[2, 1], [1, 2]]), np.array([[1.2, 4], [2, 1]]))
        ),
    ],
)
@_test_numerics_supported_backends()
def test_solve_continuous_lyapunov(argument_a, argument_b, expected_output):
    output = nx.linalg.solve_continuous_lyapunov(argument_a, argument_b)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([[1]]), np.linalg.svd(np.array([[1]]))),
        (np.array([[2, 1], [1, 2]]), np.linalg.svd(np.array([[2, 1], [1, 2]]))),
    ],
)
@_test_numerics_supported_backends()
def test_svd(argument, expected_output):
    u, s, vh = nx.linalg.svd(argument)
    assert nx.allclose(u, expected_output[0])
    assert nx.allclose(s, expected_output[1])
    assert nx.allclose(vh, expected_output[2])
