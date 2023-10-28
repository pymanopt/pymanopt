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
    "polyfit",
    "polyval",
    "prod",
    "random",
    "real",
    "sin",
    "sinc",
    "sort",
    "spacing",
    "sqrt",
    "sum",
    "tile",
    "trace",
    "transpose",
    "tanh",
    "tensordot",
    "vectorize",
    "vstack",
    "where",
    "zeros"
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
    polyfit,
    polyval,
    prod,
    real,
    sin,
    sinc,
    sort,
    spacing,
    sqrt,
    sum,
    tile,
    trace,
    transpose,
    tanh,
    tensordot,
    vectorize,
    vstack,
    where,
    zeros
)
from pymanopt.numerics import linalg
from numpy import random

_BACKENDS = ["numpy", "jax", "pytorch", "tensorflow"]


def register_backends():
    for backend in _BACKENDS:
        try:
            importlib.import_module(f"pymanopt.numerics._backends.{backend}")
        except ImportError:
            pass


register_backends()
