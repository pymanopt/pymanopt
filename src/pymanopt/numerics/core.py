import functools
import numpy as np
import torch
import jax
import tensorflow as tf


def _not_implemented(function):
    @functools.wraps(function)
    def inner(*arguments):
        raise TypeError(
            f"Function '{function.__name__}' not implemented for arguments of "
            f"type '{type(arguments[0])}'"
        )

    return inner


@functools.singledispatch
@_not_implemented
def abs(_):
    pass


@functools.singledispatch
@_not_implemented
def all(_):
    pass


@functools.singledispatch
@_not_implemented
def allclose(_):
    pass


@functools.singledispatch
@_not_implemented
def any(_):
    pass


arange = np.arange


@functools.singledispatch
@_not_implemented
def arccos(_):
    pass


@functools.singledispatch
@_not_implemented
def arccosh(_):
    pass


@functools.singledispatch
@_not_implemented
def arctan(_):
    pass


@functools.singledispatch
@_not_implemented
def arctanh(_):
    pass


@functools.singledispatch
@_not_implemented
def argmin(_):
    pass


@functools.singledispatch
@_not_implemented
def array(_):
    pass


@functools.singledispatch
@_not_implemented
def block(_):
    pass


@functools.singledispatch
@_not_implemented
def conjugate(_):
    pass


@functools.singledispatch
@_not_implemented
def cos(_):
    pass


@functools.singledispatch
@_not_implemented
def diag(_):
    pass


@functools.singledispatch
@_not_implemented
def diagonal(_):
    pass


eye = np.eye


@functools.singledispatch
@_not_implemented
def exp(_):
    pass


@functools.singledispatch
@_not_implemented
def expand_dims(_):
    pass


finfo = np.finfo


float64 = np.float64


@functools.singledispatch
@_not_implemented
def hstack(_):
    pass


integer = np.integer


@functools.singledispatch
@_not_implemented
def iscomplexobj(_):
    pass


@functools.singledispatch
@_not_implemented
def isnan(_):
    pass


@functools.singledispatch
@_not_implemented
def isrealobj(_):
    pass


@functools.singledispatch
@_not_implemented
def log(_):
    pass


logspace = np.logspace


ndarray = np.ndarray | torch.Tensor | jax.numpy.ndarray | tf.Tensor


newaxis = None


ones = np.ones


pi = np.pi


@functools.singledispatch
@_not_implemented
def polyfit(_):
    pass


@functools.singledispatch
@_not_implemented
def polyval(_):
    pass


@functools.singledispatch
@_not_implemented
def prod(_):
    pass


@functools.singledispatch
@_not_implemented
def real(_):
    pass


@functools.singledispatch
@_not_implemented
def sin(_):
    pass


@functools.singledispatch
@_not_implemented
def sinc(_):
    pass


@functools.singledispatch
@_not_implemented
def sort(_):
    pass


@functools.singledispatch
@_not_implemented
def spacing(_):
    pass


@functools.singledispatch
@_not_implemented
def sqrt(_):
    pass


@functools.singledispatch
@_not_implemented
def sum(_):
    pass


@functools.singledispatch
@_not_implemented
def tile(_):
    pass


@functools.singledispatch
@_not_implemented
def trace(_):
    pass


@functools.singledispatch
@_not_implemented
def transpose(_):
    pass


@functools.singledispatch
@_not_implemented
def tanh(_):
    pass


@functools.singledispatch
@_not_implemented
def tensordot(_):
    pass


triu_indices = np.triu_indices


vectorize = np.vectorize


@functools.singledispatch
@_not_implemented
def vstack(_):
    pass


@functools.singledispatch
@_not_implemented
def where(_):
    pass


zeros = np.zeros
