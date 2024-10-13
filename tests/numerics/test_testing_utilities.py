from contextlib import nullcontext

import jax.numpy as jnp
import numpy as np
import numpy.testing as np_testing
import pytest
import tensorflow as tf
import torch

from pymanopt.numerics import (
    JaxNumericsBackend,
    NumpyNumericsBackend,
    PytorchNumericsBackend,
    TensorflowNumericsBackend,
)


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
    "input",
    [
        [-1.0, 2.0],
        [-1, 2],
        (-1.0, 2.0),
        np.array([-1.0, 2.0], dtype=np.float64),
        np.array([-1.0, 2.0], dtype=np.float32),
        torch.tensor([-1.0, 2.0], dtype=torch.float64),
        torch.tensor([-1.0, 2.0], dtype=torch.float32),
    ],
)
@pytest.mark.parametrize(
    "expected_output, backend",
    [
        (
            np.array([-1.0, 2.0], dtype=np.float64),
            NumpyNumericsBackend(dtype=np.float64),
        ),
        (
            np.array([-1.0, 2.0], dtype=np.float32),
            NumpyNumericsBackend(dtype=np.float32),
        ),
        (
            torch.tensor([-1.0, 2.0], dtype=torch.float64),
            PytorchNumericsBackend(dtype=torch.float64),
        ),
        (
            torch.tensor([-1.0, 2.0], dtype=torch.float32),
            PytorchNumericsBackend(dtype=torch.float32),
        ),
    ],
)
def test_convert_array_to_backend(input, expected_output, backend):
    result = backend.array(input)
    assert type(result) is type(expected_output)
    assert result.dtype == expected_output.dtype
    np_testing.assert_array_equal(result, expected_output)


@pytest.mark.parametrize(
    "input1, input2, expectation",
    [
        ([-1.0, 2.0], [-1.0, 2.0], nullcontext()),
        ([-1.0, 2.0], [-1.0, 2.1], pytest.raises(AssertionError)),
        ([-1.0, 2.0], [-1.1, 2.0], pytest.raises(AssertionError)),
        (1.0, 1.000001, pytest.raises(AssertionError)),
        (np.nan, np.nan, nullcontext()),
        (np.inf, np.inf, nullcontext()),
    ],
)
@pytest.mark.parametrize("backend", all_backends)
def test_assert_allclose(input1, input2, expectation, backend):
    input1 = backend.array(input1)
    input2 = backend.array(input2)
    with expectation:
        backend.assert_allclose(input1, input2)


# @pytest.mark.parametrize(
#     "input1, input2, expectation",
#     [
#         (1.0, 2.0, pytest.raises(AssertionError)),
#         (1.0, 1.000001, pytest.raises(AssertionError)),
#         (1.0, 1.0000001, nullcontext()),
#         (np.nan, np.nan, nullcontext()),
#         (np.inf, np.inf, nullcontext()),
#     ],
# )
# @pytest.mark.parametrize("backend", all_backends)
# def test_assert_almost_equal(input1, input2, expectation, backend):
#     with expectation:
#         backend.assert_almost_equal(input1, input2)
#
#
# @pytest.mark.parametrize(
#     "input1, input2, expectation",
#     [
#         ([-1.0, 2.0], [-1.0, 2.0], nullcontext()),
#         ([-1.0, 2.0], [-1.0, 2.1], pytest.raises(AssertionError)),
#         ([-1.0, 2.0], [-1.1, 2.0], pytest.raises(AssertionError)),
#     ],
# )
# @pytest.mark.parametrize("backend", all_backends)
# def test_assert_array_almost_equal(input1, input2, expectation, backend):
#     input1 = backend.array(input1)
#     input2 = backend.array(input2)
#     with expectation:
#         backend.assert_array_almost_equal(input1, input2)
