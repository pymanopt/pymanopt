__all__ = [
    "abs",
    "all",
    "allclose",
    "any",
    "arange",
    "arccos",
    "arccosh",
    "arctan",
    "arctanh",
    "argmin",
    "array",
    "block",
    "conjugate",
    "cos",
    "diag",
    "diagonal",
    "exp",
    "expand_dims",
    "eye",
    "finfo",
    "float64",
    "hstack",
    "integer",
    "iscomplexobj",
    "isnan",
    "isrealobj",
    "linalg",
    "log",
    "logspace",
    "ndarray",
    "newaxis",
    "ones",
    "pi",
    "prod",
    "random",
    "real",
    "seterr",
    "sin",
    "sinc",
    "sort",
    "spacing",
    "special",
    "sqrt",
    "squeeze",
    "sum",
    "tile",
    "trace",
    "transpose",
    "triu_indices",
    "tanh",
    "tensordot",
    "where",
    "zeros",
    "zeros_like",
]


import importlib
from typing import Union

from pymanopt.numerics.core import (
    abs,
    all,
    allclose,
    any,
    arange,
    arccos,
    arccosh,
    arctan,
    arctanh,
    argmin,
    array,
    block,
    conjugate,
    cos,
    diag,
    diagonal,
    exp,
    expand_dims,
    eye,
    finfo,
    float64,
    hstack,
    integer,
    iscomplexobj,
    isnan,
    isrealobj,
    log,
    logspace,
    ndarray,
    newaxis,
    ones,
    pi,
    prod,
    real,
    seterr,
    sin,
    sinc,
    sort,
    spacing,
    sqrt,
    squeeze,
    sum,
    tile,
    trace,
    transpose,
    triu_indices,
    tanh,
    tensordot,
    where,
    zeros,
    zeros_like,
)
from pymanopt.numerics import linalg, special
from numpy import random

NUMERICS_SUPPORTED_BACKENDS = ["numpy", "pytorch"]


def register_backends():
    for backend in NUMERICS_SUPPORTED_BACKENDS:
        try:
            importlib.import_module(f"pymanopt.numerics._backends.{backend}")
        except ImportError:
            pass


def numpy_to_backend(point, backend):
    if isinstance(point, Union[tuple, list]):
        return tuple(numpy_to_backend(p, backend) for p in point)

    if point.dtype.kind == 'c':
        dtype = 'complex128'
    else:
        dtype = 'float64'

    if backend == 'numpy':
        import numpy as np
        point = np.array(point, dtype=dtype)
    elif backend == 'pytorch':
        import torch
        point = torch.tensor(point, dtype=getattr(torch, dtype))
    elif backend == 'jax':
        import jax.numpy as jnp
        point = jnp.array(point, dtype=dtype)
    elif backend == 'tensorflow':
        import tensorflow as tf
        point = tf.convert_to_tensor(point, dtype=getattr(tf, dtype))
    else:
        raise ValueError(f"Unknown backend '{backend}'")

    return point


register_backends()
