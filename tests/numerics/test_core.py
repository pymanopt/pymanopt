import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch

import pymanopt.numerics as nx


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([-4, 2]), np.array([4, 2])),
        (jnp.array([-4, 2]), jnp.array([4, 2])),
        (torch.Tensor([-4, 2]), torch.Tensor([4, 2])),
        (tf.constant([-4, 2]), tf.constant([4, 2])),
    ],
)
def test_abs(argument, expected_output):
    output = nx.abs(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([True, True]), True),
        (np.array([False, True]), False),
        (jnp.array([True, True]), True),
        (jnp.array([False, True]), False),
        (torch.Tensor([True, True]), True),
        (torch.Tensor([False, True]), False),
        (tf.constant([True, True]), True),
        (tf.constant([False, True]), False),
    ],
)
def test_all(argument, expected_output):
    output = nx.all(argument)
    assert output == expected_output


@pytest.mark.parametrize(
    "argument_a, argument_b, expected_output",
    [
        (4, 4, True),
        (4.2, 4.2, True),
        (np.array(4.2), 4.2, True),
        (np.array([4, 2]), np.array([4, 2]), True),
        (np.array([4, 2]), np.array([2, 4]), False),
        (jnp.array([4, 2]), jnp.array([4, 2]), True),
        (jnp.array([4, 2]), jnp.array([2, 4]), False),
        (torch.Tensor([4, 2]), torch.Tensor([4, 2]), True),
        (torch.Tensor([4, 2]), torch.Tensor([2, 4]), False),
        (tf.constant([4, 2]), tf.constant([4, 2]), True),
        (tf.constant([4, 2]), tf.constant([2, 4]), False),
    ],
)
def test_allclose(argument_a, argument_b, expected_output):
    assert nx.allclose(argument_a, argument_b) == expected_output


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        ([True, False], True),
        ([False, False], False),
        (np.array([True, False]), True),
        (np.array([False, False]), False),
    ],
)
def test_any(argument, expected_output):
    assert nx.any(argument) == expected_output


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (3, np.arange(3)),
        ((1, 3), np.arange(1, 3)),
    ],
)
def test_arange(argument, expected_output):
    if isinstance(argument, tuple):
        output = nx.arange(*argument)
    else:
        output = nx.arange(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (0, np.arccos(0)),
        (0.2, np.arccos(0.2)),
        (np.array([-0.2, 0.3]), np.arccos([-0.2, 0.3])),
    ]
)
def test_arccos(argument, expected_output):
    output = nx.arccos(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (1, np.arccosh(1)),
        (1.2, np.arccosh(1.2)),
        (np.array([1.2, 2]), np.arccosh([1.2, 2])),
    ]
)
def test_arccosh(argument, expected_output):
    output = nx.arccosh(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (1, np.arctan(1)),
        (1.2, np.arctan(1.2)),
        (np.array([1.2, 2]), np.arctan([1.2, 2])),
    ]
)
def test_arctan(argument, expected_output):
    output = nx.arctan(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (0, np.arctanh(0)),
        (0.1, np.arctanh(0.1)),
        (np.array([0.2, -0.3]), np.arctanh([0.2, -0.3])),
    ]
)
def test_arctanh(argument, expected_output):
    output = nx.arctanh(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        ([2, 1, 3], 1),
        (np.array([2, 1, 3]), 1),
    ]
)
def test_argmin(argument, expected_output):
    assert nx.argmin(argument) == expected_output


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (1, np.array([1])),
        (1.2, np.array([1.2])),
        (np.array([1, 2]), np.array([1, 2])),
    ]
)
def test_array(argument, expected_output):
    output = nx.array(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        ([1, 2], np.array([1, 2])),
        (np.array([1, 2]), np.array([1, 2])),
        ([np.array([1, 2]), np.array([3, 4])],
         np.block([np.array([1, 2]), np.array([3, 4])])),
    ]
)
def test_block(argument, expected_output):
    output = nx.block(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (1 + 2j, 1 - 2j),
        (np.array([1 + 2j, 3 + 4j]), np.array([1 - 2j, 3 - 4j])),
    ]
)
def test_conjugate(argument, expected_output):
    output = nx.conjugate(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (1, np.cos(1)),
        (1.2, np.cos(1.2)),
        (np.array([1.2, 2]), np.cos([1.2, 2]))
    ]
)
def test_cos(argument, expected_output):
    output = nx.cos(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([[1, 2], [3, 4]]), np.array([1, 4])),
    ]
)
def test_diag(argument, expected_output):
    output = nx.diag(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([[1, 2], [3, 4]]), np.diagonal(np.array([[1, 2], [3, 4]]))),
    ]
)
def test_diagonal(argument, expected_output):
    output = nx.diagonal(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (1, np.eye(1)),
        (2, np.eye(2)),
    ]
)
def test_eye(argument, expected_output):
    output = nx.eye(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.log(np.array([4, 2])), np.array([4, 2])),
        (jnp.log(jnp.array([4, 2])), jnp.array([4, 2])),
        (torch.log(torch.Tensor([4, 2])), torch.Tensor([4, 2])),
        (tf.math.log(tf.constant([4.0, 2.0])), tf.constant([4.0, 2.0])),
    ],
)
def test_exp(argument, expected_output):
    output = nx.exp(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([[1]]), np.array([[1]])),
        (np.array([[2, 1], [1, 2]]), np.linalg.cholesky(np.array([[2, 1], [1, 2]]))),
    ],
)
def test_cholesky(argument, expected_output):
    output = nx.linalg.cholesky(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument_a, argument_b, expected_output",
    [
        (np.array([-4, 2]), np.array([1, 3]), 2),
        (jnp.array([-4, 2]), jnp.array([1, 3]), 2),
        (torch.Tensor([-4, 2]), torch.Tensor([1, 3]), 2),
        (tf.constant([-4, 2]), tf.constant([1, 3]), 2),
    ],
)
def test_tensordot(argument_a, argument_b, expected_output):
    output = nx.tensordot(argument_a, argument_b, axes=argument_a.ndim)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.arctanh(np.array([0.4, 0.2])), np.array([0.4, 0.2])),
        (jnp.arctanh(jnp.array([0.4, 0.2])), jnp.array([0.4, 0.2])),
        (torch.arctanh(torch.Tensor([0.4, 0.2])), torch.Tensor([0.4, 0.2])),
        (tf.math.atanh(tf.constant([0.4, 0.2])), tf.constant([0.4, 0.2])),
    ],
)
def test_tanh(argument, expected_output):
    output = nx.tanh(argument)
    assert nx.allclose(output, expected_output)
