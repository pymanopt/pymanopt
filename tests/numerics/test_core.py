import numpy as np
import pytest
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
    "argument_a, argument_b, expected_output",
    [
        (np.array([4, 2]), 0, np.expand_dims(np.array([4, 2]), axis=0)),
        (np.array([4, 2]), (0, 1), np.expand_dims(np.array([4, 2]), axis=(0, 1))),
    ]
)
@_test_numerics_supported_backends
def test_expand_dims(argument_a, argument_b, expected_output):
    output = nx.expand_dims(argument_a, argument_b)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument",
    [
        np.float64,
        np.complex128,
    ]
)
@_test_numerics_supported_backends
def test_finfo(argument):
    assert nx.finfo(argument) == np.finfo(argument)


def test_float64():
    assert nx.float64 == np.float64


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (1, False),
        (1 + 2j, True),
        (np.array([1 + 1j, 2]), True),
    ]
)
@_test_numerics_supported_backends
def test_iscomplexobj(argument, expected_output):
    output = nx.iscomplexobj(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.nan, True),
        (np.array([np.nan, 1]), [True, False]),
        (np.array([1, 2]), [False, False]),
    ]
)
@_test_numerics_supported_backends
def test_isnan(argument, expected_output):
    output = nx.isnan(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (3, np.log(3)),
        (np.array([1, 2]), np.log(np.array([1, 2]))),
    ]
)
@_test_numerics_supported_backends
def test_log(argument, expected_output):
    output = nx.log(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        ((3, 6, 8), np.logspace(3, 6, 8)),
    ]
)
@_test_numerics_supported_backends
def test_logspace(argument, expected_output):
    output = nx.logspace(*argument)
    assert nx.allclose(output, expected_output)


def test_newaxis():
    assert nx.newaxis == np.newaxis


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (1, np.ones(1)),
        ((3, 2), np.ones((3, 2))),
    ]
)
@_test_numerics_supported_backends
def test_ones(argument, expected_output):
    output = nx.ones(argument)
    assert nx.allclose(output, expected_output)


def test_pi():
    assert nx.pi == np.pi


@pytest.mark.parametrize(
    "argument_a, argument_b, expected_output",
    [
        (
            np.array([1.2, 2]),
            np.array([3, 4]),
            np.polyfit(
                np.array([1.2, 2]), np.array([3, 4]), 1
            )
        ),
    ]
)
@_test_numerics_supported_backends
def test_polyfit(argument_a, argument_b, expected_output):
    output = nx.polyfit(argument_a, argument_b, 1)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument_a, argument_b, expected_output",
    [
        (
            np.array([1.2, 2]),
            np.array([3, 4]),
            np.polyval(np.array([1.2, 2]), np.array([3, 4]))
        ),
    ]
)
@_test_numerics_supported_backends
def test_polyval(argument_a, argument_b, expected_output):
    output = nx.polyval(argument_a, argument_b)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.array([1.2, 2]), 2.4),
    ]
)
@_test_numerics_supported_backends
def test_prod(argument, expected_output):
    output = nx.prod(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (1, 1),
        (1 + 2j, 1),
        (np.array([1 + 1j, 2]), np.array([1, 2])),
    ]
)
@_test_numerics_supported_backends
def test_real(argument, expected_output):
    output = nx.real(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (2, np.sin(2)),
        (np.array([1, 2]), np.sin(np.array([1, 2]))),
    ]
)
@_test_numerics_supported_backends
def test_sin(argument, expected_output):
    output = nx.sin(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (2, np.sinc(2)),
        (np.array([1, 2]), np.sinc(np.array([1, 2]))),
    ]
)
@_test_numerics_supported_backends
def test_sinc(argument, expected_output):
    output = nx.sinc(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        ([2, 1], np.sort([2, 1])),
        (np.array([2, 1]), np.sort(np.array([2, 1]))),
    ]
)
@_test_numerics_supported_backends
def test_sort(argument, expected_output):
    output = nx.sort(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (2.2, np.spacing(2.2)),
    ]
)
@_test_numerics_supported_backends
def test_spacing(argument, expected_output):
    output = nx.spacing(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (2, np.sqrt(2)),
        ([2, 1], np.sqrt([2, 1])),
        (np.array([2, 1]), np.sqrt(np.array([2, 1]))),
    ]
)
@_test_numerics_supported_backends
def test_sqrt(argument, expected_output):
    output = nx.sqrt(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument_a, argument_b, expected_output",
    [
        (1, None, 1),
        ([1, 2], None, 3),
        (np.array([1, 2]), None, 3),
        (np.array([[1, 2], [3, 4]]), 1, np.sum(np.array([[1, 2], [3, 4]]), 1)),
    ]
)
@_test_numerics_supported_backends
def test_sum(argument_a, argument_b, expected_output):
    if argument_b is None:
        output = nx.sum(argument_a)
    else:
        output = nx.sum(argument_a, argument_b)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument_a, argument_b, expected_output",
    [
        (1, [1, 2], np.tile(1, [1, 2])),
        ([1, 2], [3, 4, 1], np.tile([1, 2], [3, 4, 1])),
        (
            np.array([1, 2]),
            np.array([3, 4, 1]),
            np.tile(np.array([1, 2]), np.array([3, 4, 1]))
        ),
    ]
)
@_test_numerics_supported_backends
def test_tile(argument_a, argument_b, expected_output):
    output = nx.tile(argument_a, argument_b)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (np.arctanh(np.array([0.4, 0.2])), np.array([0.4, 0.2])),
    ],
)
@_test_numerics_supported_backends
def test_tanh(argument, expected_output):
    output = nx.tanh(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument_a, argument_b, expected_output",
    [
        (np.array([-4, 2]), np.array([1, 3]), 2),
    ],
)
@_test_numerics_supported_backends
def test_tensordot(argument_a, argument_b, expected_output):
    output = nx.tensordot(argument_a, argument_b, axes=argument_a.ndim)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        ([[1, 2], [3, 4]], np.trace([[1, 2], [3, 4]])),
        (np.array([[1, 2], [3, 4]]), np.trace(np.array([[1, 2], [3, 4]]))),
    ]
)
@_test_numerics_supported_backends
def test_trace(argument, expected_output):
    output = nx.trace(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument_a, argument_b, expected_output",
    [
        (2, None, 2),
        ([1, 2], None, np.transpose([1, 2])),
        (np.array([1, 2]), None, np.transpose(np.array([1, 2]))),
        (
            np.array([[1, 2], [3, 4]]),
            (1, 0),
            np.transpose(np.array([[1, 2], [3, 4]]), (1, 0))
        ),
    ]
)
@_test_numerics_supported_backends
def test_transpose(argument_a, argument_b, expected_output):
    if argument_b is None:
        output = nx.transpose(argument_a)
    else:
        output = nx.transpose(argument_a, argument_b)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument_a, argument_b, argument_c, expected_output",
    [
        (1, 2, None, np.triu_indices(1, 2)),
        (2, 1, 1, np.triu_indices(2, 1, 1)),
    ]
)
@_test_numerics_supported_backends
def test_triu_indices(argument_a, argument_b, argument_c, expected_output):
    if argument_c is None:
        output = nx.triu_indices(argument_a, argument_b)
    else:
        output = nx.triu_indices(argument_a, argument_b, argument_c)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument_a, argument_b, argument_c, expected_output",
    [
        (
            scipy.linalg.expm,
            "(m,m)->(m,m)",
            [np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])],
            np.array([
                scipy.linalg.expm(np.array([[1, 2], [3, 4]])) for _ in range(2)
            ])
        ),
    ]
)
@_test_numerics_supported_backends
def test_vectorize(argument_a, argument_b, argument_c, expected_output):
    output = nx.vectorize(argument_a, signature=argument_b)
    assert nx.allclose(output(argument_c), expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (False, False),
        ([False, True], np.where([False, True])),
        (np.array([False, True]), np.where(np.array([False, True]))),
    ]
)
@_test_numerics_supported_backends
def test_where(argument, expected_output):
    output = nx.where(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (2, np.zeros(2)),
        ([2, 1], np.zeros([2, 1])),
    ]
)
@_test_numerics_supported_backends
def test_zeros(argument, expected_output):
    output = nx.zeros(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output",
    [
        (2, np.zeros_like(2)),
        ([2, 1], np.zeros_like([2, 1])),
    ]
)
@_test_numerics_supported_backends
def test_zeros_like(argument, expected_output):
    output = nx.zeros_like(argument)
    assert nx.allclose(output, expected_output)
