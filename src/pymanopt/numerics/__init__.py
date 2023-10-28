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
    "eye",
    "exp",
    "finfo",
    "hstack",
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
    eye,
    exp,
    finfo,
    hstack,
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
from pymanopt.numerics import random


def register_backends():
    for backend in ["numpy", "jax", "pytorch", "tensorflow"]:
        try:
            importlib.import_module(f"pymanopt.numerics._backends.{backend}")
        except ImportError:
            pass


register_backends()
