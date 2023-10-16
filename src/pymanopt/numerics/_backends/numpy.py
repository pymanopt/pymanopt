import numpy as np

import pymanopt.numerics.core as nx

generic_np_type = int | float | np.generic | np.ndarray


@nx.abs.register
def _(array: np.ndarray) -> np.ndarray:
    return np.abs(array)


@nx.all.register
def _(array: np.ndarray) -> bool:
    return np.all(array)


@nx.allclose.register
def _(array_a: generic_np_type, array_b: generic_np_type) -> bool:
    return np.allclose(array_a, array_b)


@nx.any.register
def _(array: list | np.ndarray) -> bool:
    return np.any(array)


@nx.arange.register
def _(start: int, stop: int = None, step: int = 1) -> np.ndarray:
    return np.arange(start, stop, step)


@nx.arccos.register
def _(array: generic_np_type) -> generic_np_type:
    return np.arccos(array)


@nx.arccosh.register
def _(array: generic_np_type) -> generic_np_type:
    return np.arccosh(array)


@nx.arctan.register
def _(array: generic_np_type) -> generic_np_type:
    return np.arctan(array)


@nx.arctanh.register
def _(array: generic_np_type) -> generic_np_type:
    return np.arctanh(array)


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
