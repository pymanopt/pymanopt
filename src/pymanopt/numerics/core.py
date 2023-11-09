import functools
import numpy as np
from typing import Sequence

from .dispatch import SequenceDispatch


def not_implemented(function):
    @functools.wraps(function)
    def inner(*arguments):
        if isinstance(arguments[0], Sequence):
            type_str = f"Sequence[{type(arguments[0][0])}]"
        else:
            type_str = str(type(arguments[0]))
        raise TypeError(
            f"Function '{function.__name__}' not implemented for arguments of "
            f"type '{type_str}'."
        )

    return inner


@functools.singledispatch
@not_implemented
def abs(_):
    pass


@functools.singledispatch
@not_implemented
def all(_):
    pass


@functools.singledispatch
@not_implemented
def allclose(_):
    pass


@functools.singledispatch
@not_implemented
def any(_):
    pass


arange = np.arange


@functools.singledispatch
@not_implemented
def arccos(_):
    pass


@functools.singledispatch
@not_implemented
def arccosh(_):
    pass


@functools.singledispatch
@not_implemented
def arctan(_):
    pass


@functools.singledispatch
@not_implemented
def arctanh(_):
    pass


@functools.singledispatch
@not_implemented
def argmin(_):
    pass


@functools.singledispatch
@not_implemented
def array(_):
    pass


@functools.singledispatch
@not_implemented
def assert_almost_equal(_):
    pass


@functools.singledispatch
@not_implemented
def assert_array_almost_equal(_):
    pass


@SequenceDispatch
@not_implemented
def block(_):
    pass


@functools.singledispatch
@not_implemented
def conjugate(_):
    pass


@functools.singledispatch
@not_implemented
def cos(_):
    pass


@functools.singledispatch
@not_implemented
def diag(_):
    pass


@functools.singledispatch
@not_implemented
def diagonal(_):
    pass


eye = np.eye


@functools.singledispatch
@not_implemented
def exp(_):
    pass


@functools.singledispatch
@not_implemented
def expand_dims(_):
    pass


finfo = np.finfo


float64 = np.float64


@SequenceDispatch
@not_implemented
def hstack(_):
    pass


integer = np.integer


@functools.singledispatch
@not_implemented
def iscomplexobj(_):
    pass


@functools.singledispatch
@not_implemented
def isnan(_):
    pass


@functools.singledispatch
@not_implemented
def isrealobj(_):
    pass


@functools.singledispatch
@not_implemented
def log(_):
    pass


logspace = np.logspace


ndarray = np.ndarray
try:
    import torch
    ndarray = ndarray | torch.Tensor
except ImportError:
    pass
try:
    import jax
    ndarray = ndarray | jax.numpy.ndarray
except ImportError:
    pass
try:
    import tensorflow as tf
    ndarray = ndarray | tf.Tensor
except ImportError:
    pass


newaxis = None


ones = np.ones


pi = np.pi


@functools.singledispatch
@not_implemented
def prod(_):
    pass


@functools.singledispatch
@not_implemented
def real(_):
    pass


def seterr(all=None):
    np.seterr(all=all)
    if all == 'raise':
        try:
            import torch
            torch.autograd.set_detect_anomaly(True)
        except ImportError:
            pass
        try:
            import jax
            jax.config.update("jax_debug_nans", True)
        except ImportError:
            pass
        try:
            import tensorflow as tf
            tf.debugging.enable_check_numerics()
        except ImportError:
            pass


@functools.singledispatch
@not_implemented
def sin(_):
    pass


@functools.singledispatch
@not_implemented
def sinc(_):
    pass


@functools.singledispatch
@not_implemented
def sort(_):
    pass


@functools.singledispatch
@not_implemented
def spacing(_):
    pass


@functools.singledispatch
@not_implemented
def sqrt(_):
    pass


@functools.singledispatch
@not_implemented
def squeeze(_):
    pass


@functools.singledispatch
@not_implemented
def sum(_):
    pass


@functools.singledispatch
@not_implemented
def tile(_):
    pass


@functools.singledispatch
@not_implemented
def trace(_):
    pass


@functools.singledispatch
@not_implemented
def transpose(_):
    pass


@functools.singledispatch
@not_implemented
def tanh(_):
    pass


@functools.singledispatch
@not_implemented
def tensordot(_):
    pass


triu_indices = np.triu_indices


@SequenceDispatch
@not_implemented
def vstack(_):
    pass


@functools.singledispatch
@not_implemented
def where(_):
    pass


zeros = np.zeros


@functools.singledispatch
@not_implemented
def zeros_like(_):
    pass
