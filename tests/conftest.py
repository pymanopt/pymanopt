import random

import autograd.numpy as anp
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest
import tensorflow as tf
import torch

from pymanopt.numerics import (  # TensorflowNumericsBackend,
    JaxNumericsBackend,
    NumericsBackend,
    NumpyNumericsBackend,
    PytorchNumericsBackend,
)


matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def initialize_test_state():
    seed = 1
    random.seed(seed)
    anp.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)


_REAL_NUMERICS_BACKENDS = [
    NumpyNumericsBackend(np.float64),
    PytorchNumericsBackend(torch.float64),
    JaxNumericsBackend(jnp.float64),
    # TensorflowNumericsBackend(tf.float64),
    NumpyNumericsBackend(np.float32),
    PytorchNumericsBackend(torch.float32),
    JaxNumericsBackend(jnp.float32),
    # TensorflowNumericsBackend(tf.float32),
]

_COMPLEX_NUMERICS_BACKENDS = [
    NumpyNumericsBackend(np.complex128),
    PytorchNumericsBackend(torch.complex128),
    JaxNumericsBackend(jnp.complex128),
    # TensorflowNumericsBackend(tf.complex128),
]


@pytest.fixture(params=_REAL_NUMERICS_BACKENDS)
def real_numerics_backend(request) -> NumericsBackend:
    return request.param


@pytest.fixture(params=_COMPLEX_NUMERICS_BACKENDS)
def complex_numerics_backend(request) -> NumericsBackend:
    return request.param


@pytest.fixture(params=_REAL_NUMERICS_BACKENDS + _COMPLEX_NUMERICS_BACKENDS)
def all_numerics_backend(request) -> NumericsBackend:
    return request.param
