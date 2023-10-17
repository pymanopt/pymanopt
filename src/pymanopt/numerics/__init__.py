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
    "log",
    "logspace",
    "newaxis",
    "ones",
    "pi",
    "polyfit",
    "polyval",
    "prod",
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
    "tensordot"
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
    tensordot
)


def register_backends():
    for backend in ["numpy", "jax", "pytorch", "tensorflow"]:
        try:
            importlib.import_module(f"pymanopt.numerics._backends.{backend}")
        except ImportError:
            pass


register_backends()
