__all__ = ["abs", "allclose", "exp", "tensordot"]


import importlib

from pymanopt.numerics.core import abs, allclose, exp, tensordot


FUNCTIONS = [abs, allclose, exp, tensordot]


def register_backends():
    for backend in ["numpy", "jax", "pytorch", "tensorflow"]:
        try:
            importlib.import_module(f"pymanopt.numerics._backends.{backend}")
        except ImportError:
            pass


register_backends()
