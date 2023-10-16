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
    "exp",
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
    exp,
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
