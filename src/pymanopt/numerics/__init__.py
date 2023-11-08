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
    "vstack",
    "where",
    "zeros",
    "zeros_like",
]


import importlib

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
    vstack,
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


def get_backend(point):
    backend = None

    while isinstance(point, (tuple, list)):
        point = point[0]

    import numpy as np
    if isinstance(point, np.ndarray):
        backend = "numpy"

    try:
        import torch
        if isinstance(point, torch.Tensor):
            backend = "pytorch"
    except ImportError:
        pass

    try:
        import jax.numpy as jnp
        if isinstance(point, jnp.ndarray):
            backend = "jax"
    except ImportError:
        pass

    try:
        import tensorflow as tf
        if isinstance(point, tf.Tensor):
            backend = "tensorflow"
    except ImportError:
        pass

    if backend is None:
        raise ValueError("Unknown backend...")

    return backend


def numpy_to_backend(point, backend):
    # if point is a namedtuple, convert each point
    if hasattr(point, '_fields'):
        point = point.__class__(*[
            numpy_to_backend(p, backend) for p in point
        ])
        return point
    if isinstance(point, tuple):
        return tuple(numpy_to_backend(p, backend) for p in point)
    if isinstance(point, list):
        return [numpy_to_backend(p, backend) for p in point]

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


def array_as(point, as_):
    backend = get_backend(as_)
    return numpy_to_backend(point, backend)


register_backends()
