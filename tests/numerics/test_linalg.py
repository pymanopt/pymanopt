import numpy as np
import pytest
import scipy

import pymanopt.numerics as nx


@pytest.mark.parametrize(
    "argument",
    [
        np.array([[1]]),
        np.array([[2, 1], [1, 2]])
    ],
)
def test_cholesky(argument):
    output = nx.linalg.cholesky(argument)
    expected_output = np.linalg.cholesky(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument",
    [
        np.array([[1]]),
        np.array([[2, 1], [1, 2]]),
    ],
)
def test_det(argument):
    output = nx.linalg.det(argument)
    expected_output = np.linalg.det(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument",
    [
        np.array([[2, 1], [1, 2]])
    ],
)
def test_eigh(argument):
    eigenvalues, eigenvectors = nx.linalg.eigh(argument)
    true_eigenvalues, true_eigenvectors = np.linalg.eigh(argument)
    assert nx.allclose(eigenvalues, true_eigenvalues)
    assert nx.allclose(eigenvectors, true_eigenvectors)


@pytest.mark.parametrize(
    "argument",
    [
        np.array([[1]]),
        np.array([[1, 1.3], [1.3, 1]]),
    ],
)
def test_expm(argument):
    output = nx.linalg.expm(argument)
    expected_output = scipy.linalg.expm(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument",
    [
        np.array([[1]]),
        np.array([[2, 1], [1, 2]]),
    ],
)
def test_inv(argument):
    output = nx.linalg.inv(argument)
    expected_output = np.linalg.inv(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument",
    [
        np.array([[1]]),
        np.array([[2, 1], [1, 2]]),
    ],
)
def test_logm(argument):
    output = nx.linalg.logm(argument)
    expected_output = scipy.linalg.logm(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument",
    [
        np.array([[1]]),
        np.array([[2, 1], [4, 2]]),
        np.array([[2, 1], [4, 1]]),
    ]
)
def test_matrix_rank(argument):
    output = nx.linalg.matrix_rank(argument)
    expected_output = np.linalg.matrix_rank(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument_a, argument_b",
    [
        (1, None),
        ([2], None),
        (np.array([1, 4.2]), None),
        (np.array([[1, 2], [3, 4.2]]), 0)
    ]
)
def test_norm(argument_a, argument_b):
    output = nx.linalg.norm(argument_a, axis=argument_b)
    if argument_b is None:
        expected_output = np.linalg.norm(argument_a)
    else:
        expected_output = np.linalg.norm(argument_a, axis=argument_b)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument",
    [
        np.array([[1]]),
        np.array([[2, 1], [1, 2]]),
    ],
)
def test_qr(argument):
    output = nx.linalg.qr(argument)
    expected_output = np.linalg.qr(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument_a, argument_b",
    [
        (np.array([[1]]), np.array([[1]])),
        (np.array([[2, 1], [1, 2]]), -np.array([[2, 1], [1, 2]])),
    ],
)
def test_solve(argument_a, argument_b):
    output = nx.linalg.solve(argument_a, argument_b)
    expected_output = np.linalg.solve(argument_a, argument_b)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument_a, argument_b",
    [
        (np.array([[1]]), np.array([[3]])),
        (np.array([[2, 1], [1, 2]]), np.array([[1.2, 4], [2, 1]])),
    ],
)
def test_solve_continuous_lyapunov(argument_a, argument_b):
    output = nx.linalg.solve_continuous_lyapunov(argument_a, argument_b)
    expected_output = scipy.linalg.solve_continuous_lyapunov(argument_a, argument_b)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument",
    [
        np.array([[1]]),
        np.array([[2, 1], [1, 2]]),
    ],
)
def test_svd(argument):
    U, S, Vh = nx.linalg.svd(argument)
    true_U, true_S, true_Vh = np.linalg.svd(argument)
    assert nx.allclose(U, true_U)
    assert nx.allclose(S, true_S)
    assert nx.allclose(Vh, true_Vh)
