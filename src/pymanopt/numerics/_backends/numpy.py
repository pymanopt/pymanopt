import numpy as np

import pymanopt.numerics.core as nx


@nx.abs.register(np.ndarray)
def _(array):
    return np.abs(array)


@nx.allclose.register(np.ndarray)
def _(array_a, array_b):
    return np.allclose(array_a, array_b)


@nx.exp.register(np.ndarray)
def _(array):
    return np.abs(array)


@nx.tensordot.register(np.ndarray)
def _(array_a, array_b, axes: int):
    return np.tensordot(array_a, array_b, axes=axes)
