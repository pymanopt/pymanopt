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
    "argument",
    [
        np.float64,
        np.complex128,
    ]
)
def test_finfo(argument):
    assert nx.finfo(argument) == np.finfo(argument)


@pytest.mark.parametrize(
    "argument",
    [
        [1, 2],
        [np.array([1, 2]), np.array([3, 4])],
    ]
)
def test_hstack(argument):
    output = nx.hstack(argument)
    assert nx.allclose(output, np.hstack(argument))


@pytest.mark.parametrize(
    "argument",
    [
        1,
        1 + 2j,
        np.array([1 + 1j, 2]),
    ]
)
def test_iscomplexobj(argument):
    output = nx.iscomplexobj(argument)
    assert nx.allclose(output, np.iscomplexobj(argument))


@pytest.mark.parametrize(
    "argument",
    [
        np.nan,
        np.array([np.nan, 1]),
        np.array([1, 2]),
    ]
)
def test_isnan(argument):
    output = nx.isnan(argument)
    assert nx.allclose(output, np.isnan(argument))


@pytest.mark.parametrize(
    "argument",
    [
        3,
        np.array([1, 2]),
    ]
)
def test_log(argument):
    output = nx.log(argument)
    assert nx.allclose(output, np.log(argument))


@pytest.mark.parametrize(
    "argument",
    [
        (3, 6, 8)
    ]
)
def test_logspace(argument):
    output = nx.logspace(*argument)
    assert nx.allclose(output, np.logspace(*argument))


def test_newaxis():
    assert nx.newaxis == np.newaxis


@pytest.mark.parametrize(
    "argument",
    [
        1,
        (3, 2)
    ]
)
def test_ones(argument):
    output = nx.ones(argument)
    assert nx.allclose(output, np.ones(argument))


def test_pi():
    assert nx.pi == np.pi


@pytest.mark.parametrize(
    "argument_a, argument_b",
    [
        (np.array([1.2, 2]), np.array([3, 4])),
    ]
)
def test_polyfit(argument_a, argument_b):
    output = nx.polyfit(argument_a, argument_b, 1)
    assert nx.allclose(output, np.polyfit(argument_a, argument_b, 1))


@pytest.mark.parametrize(
    "argument_a, argument_b",
    [
        (np.array([1.2, 2]), np.array([3, 4])),
    ]
)
def test_polyval(argument_a, argument_b):
    output = nx.polyval(argument_a, argument_b)
    assert nx.allclose(output, np.polyval(argument_a, argument_b))


@pytest.mark.parametrize(
    "argument",
    [
        np.array([1.2, 2]),
    ]
)
def test_prod(argument):
    output = nx.prod(argument)
    assert nx.allclose(output, np.prod(argument))


@pytest.mark.parametrize(
    "argument",
    [
        1,
        1 + 2j,
        np.array([1 + 1j, 2]),
    ]
)
def test_real(argument):
    output = nx.real(argument)
    assert nx.allclose(output, np.real(argument))


@pytest.mark.parametrize(
    "argument",
    [
        2,
        np.array([1, 2]),
    ]
)
def test_sin(argument):
    output = nx.sin(argument)
    assert nx.allclose(output, np.sin(argument))


@pytest.mark.parametrize(
    "argument",
    [
        2,
        np.array([1, 2]),
    ]
)
def test_sinc(argument):
    output = nx.sinc(argument)
    assert nx.allclose(output, np.sinc(argument))


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
