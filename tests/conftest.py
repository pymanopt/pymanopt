import random

import autograd.numpy as anp
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest
import tensorflow as tf
import torch

from pymanopt.backends import Backend
from pymanopt.backends.autograd_backend import AutogradBackend
from pymanopt.backends.jax_backend import JaxBackend
from pymanopt.backends.numpy_backend import NumpyBackend
from pymanopt.backends.pytorch_backend import PytorchBackend


# from pymanopt.backends.tensorflow_backend import TensorflowBackend


matplotlib.use("Agg")

torch.autograd.set_detect_anomaly(True)


@pytest.fixture(autouse=True)
def initialize_test_state():
    seed = 1
    random.seed(seed)
    anp.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)


_REAL_NUMERICS_BACKENDS = [
    NumpyBackend(np.float64),
    AutogradBackend(anp.float64),
    PytorchBackend(torch.float64),
    JaxBackend(jnp.float64),
    # TensorflowBackend(tf.float64),
    NumpyBackend(np.float32),
    AutogradBackend(anp.float32),
    PytorchBackend(torch.float32),
    JaxBackend(jnp.float32),
    # TensorflowBackend(tf.float32),
]

_COMPLEX_NUMERICS_BACKENDS = [
    NumpyBackend(np.complex128),
    AutogradBackend(anp.complex128),
    PytorchBackend(torch.complex128),
    JaxBackend(jnp.complex128),
    # TensorflowBackend(tf.complex128),
]


@pytest.fixture(params=_REAL_NUMERICS_BACKENDS)
def real_numerics_backend(request) -> Backend:
    return request.param


@pytest.fixture(params=_COMPLEX_NUMERICS_BACKENDS)
def complex_numerics_backend(request) -> Backend:
    return request.param


@pytest.fixture(params=_REAL_NUMERICS_BACKENDS + _COMPLEX_NUMERICS_BACKENDS)
def all_numerics_backend(request) -> Backend:
    return request.param


@pytest.fixture(params=[1, 3])
def product_dimension(request) -> int:
    return request.param
