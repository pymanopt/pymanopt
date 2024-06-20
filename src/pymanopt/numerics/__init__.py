__all__ = ["abs", "allclose", "exp", "tensordot", "tanh"]


import importlib

from pymanopt.numerics.core import abs, allclose, exp, tanh, tensordot


def register_backends():
    for backend in ["numpy", "jax", "pytorch", "tensorflow"]:
        try:
            importlib.import_module(f"pymanopt.numerics._backends.{backend}")
        except ImportError:
            pass


register_backends()
