import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch
import scipy

import pymanopt.numerics as nx
from tests.numerics import _test_numerics_supported_backends


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([-4, 2]), np.array([4, 2])),
    ],
)
@_test_numerics_supported_backends
def test_abs(argument, expected_output):
    output = nx.abs(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([True, True]), True),
        (np.array([False, True]), False),
    ],
)
@_test_numerics_supported_backends
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
    ],
)
@_test_numerics_supported_backends
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
@_test_numerics_supported_backends
def test_any(argument, expected_output):
    assert nx.any(argument) == expected_output


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (3, np.arange(3)),
        ((1, 3), np.arange(1, 3)),
    ],
)
@_test_numerics_supported_backends
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
@_test_numerics_supported_backends
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
@_test_numerics_supported_backends
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
@_test_numerics_supported_backends
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
@_test_numerics_supported_backends
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
@_test_numerics_supported_backends
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
@_test_numerics_supported_backends
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
@_test_numerics_supported_backends
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
@_test_numerics_supported_backends
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
@_test_numerics_supported_backends
def test_cos(argument, expected_output):
    output = nx.cos(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([[1, 2], [3, 4]]), np.array([1, 4])),
    ]
)
@_test_numerics_supported_backends
def test_diag(argument, expected_output):
    output = nx.diag(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([[1, 2], [3, 4]]), np.diagonal(np.array([[1, 2], [3, 4]]))),
    ]
)
@_test_numerics_supported_backends
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
@_test_numerics_supported_backends
def test_eye(argument, expected_output):
    output = nx.eye(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.log(np.array([4, 2])), np.array([4, 2])),
    ],
)
@_test_numerics_supported_backends
def test_exp(argument, expected_output):
    output = nx.exp(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument_a, argument_b",
    [
        (np.array([4, 2]), 0),
        (np.array([4, 2]), (0, 1)),
    ]
)
def test_expand_dims(argument_a, argument_b):
    output = nx.expand_dims(argument_a, argument_b)
    assert nx.allclose(output, np.expand_dims(argument_a, argument_b))


@pytest.mark.parametrize(
    "argument",
    [
        np.float64,
        np.complex128,
    ]
)
def test_finfo(argument):
    assert nx.finfo(argument) == np.finfo(argument)


def test_float64():
    assert nx.float64 == np.float64


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
    "argument",
    [
        [2, 1],
        np.array([2, 1]),
    ]
)
def test_sort(argument):
    output = nx.sort(argument)
    assert nx.allclose(output, np.sort(argument))


@pytest.mark.parametrize(
    "argument",
    [
        2.2,
        [2.2, 1],
        np.array([2.2, 1]),
    ]
)
def test_spacing(argument):
    output = nx.spacing(argument)
    assert nx.allclose(output, np.spacing(argument))


@pytest.mark.parametrize(
    "argument",
    [
        2,
        [2, 1],
        np.array([2, 1]),
    ]
)
def test_sqrt(argument):
    output = nx.sqrt(argument)
    assert nx.allclose(output, np.sqrt(argument))


@pytest.mark.parametrize(
    "argument_a, argument_b",
    [
        (1, None),
        ([1, 2], None),
        (np.array([1, 2]), None),
        (np.array([[1, 2], [3, 4]]), 1),
    ]
)
def test_sum(argument_a, argument_b):
    if argument_b is None:
        output = nx.sum(argument_a)
        assert nx.allclose(output, np.sum(argument_a))
    else:
        output = nx.sum(argument_a, argument_b)
        assert nx.allclose(output, np.sum(argument_a, argument_b))


@pytest.mark.parametrize(
    "argument_a, argument_b",
    [
        (1, [1, 2]),
        ([1, 2], [3, 4, 1]),
        (np.array([1, 2]), np.array([3, 4, 1])),
    ]
)
def test_tile(argument_a, argument_b):
    output = nx.tile(argument_a, argument_b)
    assert nx.allclose(output, np.tile(argument_a, argument_b))


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
    "argument",
    [
        [[1, 2], [3, 4]],
        np.array([[1, 2], [3, 4]]),
    ]
)
def test_trace(argument):
    output = nx.trace(argument)
    assert nx.allclose(output, np.trace(argument))


@pytest.mark.parametrize(
    "argument_a, argument_b",
    [
        (2, None),
        ([1, 2], None),
        (np.array([1, 2]), None),
        (np.array([[1, 2], [3, 4]]), (1, 0)),
    ]
)
def test_transpose(argument_a, argument_b):
    if argument_b is None:
        output = nx.transpose(argument_a)
        assert nx.allclose(output, np.transpose(argument_a))
    else:
        output = nx.transpose(argument_a, argument_b)
        assert nx.allclose(output, np.transpose(argument_a, argument_b))


@pytest.mark.parametrize(
    "argument_a, argument_b, argument_c",
    [
        (1, 2, None),
        (2, 1, 1),
    ]
)
def test_triu_indices(argument_a, argument_b, argument_c):
    if argument_c is None:
        output = nx.triu_indices(argument_a, argument_b)
        assert nx.allclose(output, np.triu_indices(argument_a, argument_b))
    else:
        output = nx.triu_indices(argument_a, argument_b, argument_c)
        assert nx.allclose(
            output,
            np.triu_indices(argument_a, argument_b, argument_c)
        )


@pytest.mark.parametrize(
    "argument_a, argument_b, argument_c",
    [
        (scipy.linalg.expm, "(m,m)->(m,m)", [
            np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])]),
    ]
)
def test_vectorize(argument_a, argument_b, argument_c):
    output = nx.vectorize(argument_a, signature=argument_b)
    assert nx.allclose(
        output(argument_c),
        np.vectorize(argument_a, signature=argument_b)(argument_c)
    )


@pytest.mark.parametrize(
    "argument",
    [
        [2, 1],
        np.array([2, 1]),
        [np.array([2, 1]), np.array([2, 1])],
    ]
)
def test_vstack(argument):
    output = nx.vstack(argument)
    assert nx.allclose(output, np.vstack(argument))


@pytest.mark.parametrize(
    "argument",
    [
        False,
        [False, True],
        np.array([False, True]),
    ]
)
def test_where(argument):
    output = nx.where(argument)
    assert nx.allclose(output, np.where(argument))


@pytest.mark.parametrize(
    "argument",
    [
        2,
        [2, 1],
    ]
)
def test_zeros(argument):
    output = nx.zeros(argument)
    assert nx.allclose(output, np.zeros(argument))


@pytest.mark.parametrize(
    "argument",
    [
        2,
        [2, 1],
    ]
)
def test_zeros_like(argument):
    output = nx.zeros_like(argument)
    assert nx.allclose(output, np.zeros_like(argument))
