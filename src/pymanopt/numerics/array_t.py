from numbers import Number
from typing import Sequence, Union

import numpy as np


__all__ = [
    "array_t",
    "SUPPORTED_NUMERICS_BACKENDS",
    "AVAILABLE_NUMERICS_BACKENDS",
]

SUPPORTED_NUMERICS_BACKENDS = ["numpy", "pytorch", "jax", "tensorflow"]
AVAILABLE_NUMERICS_BACKENDS = ["numpy"]

array_t = Union[
    Number,
    Sequence[Number],
    Sequence[Sequence[Number]],
    Sequence[Sequence[Sequence[Number]]],
    np.ndarray,
    type(None),
]

try:
    import torch

    AVAILABLE_NUMERICS_BACKENDS.append("pytorch")
    array_t = Union[array_t, torch.Tensor]
except ImportError:
    pass

try:
    import jax

    AVAILABLE_NUMERICS_BACKENDS.append("jax")
    array_t = Union[array_t, jax.numpy.ndarray]
except ImportError:
    pass

try:
    import tensorflow as tf

    AVAILABLE_NUMERICS_BACKENDS.append("tensorflow")
    array_t = Union[array_t, tf.Tensor]
except ImportError:
    pass
