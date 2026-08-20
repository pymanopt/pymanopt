from contextlib import nullcontext as does_not_raise

import jax.numpy as jnp
import numpy as np
import numpy.testing as np_testing
import pytest
import torch

from pymanopt.backends import Backend
from pymanopt.backends.jax_backend import JaxBackend
from pymanopt.backends.numpy_backend import NumpyBackend
from pymanopt.backends.pytorch_backend import PytorchBackend


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
            NumpyBackend(dtype=np.float64),
        ),
        (
            np.array([-1.0, 2.0], dtype=np.float32),
            NumpyBackend(dtype=np.float32),
        ),
        (
            torch.tensor([-1.0, 2.0], dtype=torch.float64),
            PytorchBackend(dtype=torch.float64),
        ),
        (
            torch.tensor([-1.0, 2.0], dtype=torch.float32),
            PytorchBackend(dtype=torch.float32),
        ),
    ],
)
def test_convert_array_to_backend(input, expected_output, backend: Backend):
    result = backend.array(input)
    assert type(result) is type(expected_output)
    assert result.dtype == expected_output.dtype
    np_testing.assert_array_equal(result, expected_output)


@pytest.mark.parametrize(
    "input1, input2, exception_context",
    [
        ([-1.0, 2.0], [-1.0, 2.0], does_not_raise()),
        ([-1.0, 2.0], [-1.0, 2.1], pytest.raises(ValueError)),
        ([-1.0, 2.0], [-1.1, 2.0], pytest.raises(ValueError)),
        (1.0, 1.0001, pytest.raises(ValueError)),
        (np.nan, np.nan, pytest.raises(ValueError)),
        (np.inf, np.inf, does_not_raise()),
    ],
)
@pytest.mark.parametrize(
    "backend",
    [
        NumpyBackend(np.float64),
        PytorchBackend(torch.float32),
        JaxBackend(jnp.float64),
    ],
)
def test_assert_allclose(input1, input2, exception_context, backend: Backend):
    input1 = backend.array(input1)
    input2 = backend.array(input2)
    with exception_context:
        backend.assert_allclose(input1, input2)
