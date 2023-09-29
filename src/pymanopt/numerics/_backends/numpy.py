import numpy as np

import pymanopt.numerics.core as nx


@nx.abs.register
def _(array: np.ndarray) -> np.ndarray:
    return np.abs(array)


@nx.all.register
def _(array: np.ndarray) -> bool:
    return np.all(array)


@nx.allclose.register
def _(array_a: np.ndarray, array_b: np.ndarray) -> bool:
    return np.allclose(array_a, array_b)


@nx.exp.register
def _(array: np.ndarray) -> np.ndarray:
    return np.exp(array)


@nx.tensordot.register
def _(
    array_a: np.ndarray, array_b: np.ndarray, *, axes: int = 2
) -> np.ndarray:
    return np.tensordot(array_a, array_b, axes=axes)


@nx.tanh.register
def _(array: np.ndarray) -> np.ndarray:
    return np.tanh(array)
