import numpy as np
import scipy

import pymanopt.numerics.core as nx
import pymanopt.numerics.linalg as linalg

generic_list_type = tuple | list | np.ndarray
generic_np_type = int | float | complex | np.generic | generic_list_type


@nx.abs.register
def _(array: generic_np_type) -> np.float64 | np.ndarray:
    return np.abs(array)


@nx.all.register
def _(array: generic_np_type) -> bool:
    return np.all(array)


@nx.allclose.register
def _(array_a: generic_np_type, array_b: generic_np_type) -> bool:
    return np.allclose(array_a, array_b)


@nx.any.register
def _(array: generic_np_type) -> bool:
    return np.any(array)


@nx.arange.register
def _(start: int, stop: int = None, step: int = 1) -> np.ndarray:
    return np.arange(start, stop, step)


@nx.arccos.register
def _(array: generic_np_type) -> np.float64 | np.ndarray:
    return np.arccos(array)


@nx.arccosh.register
def _(array: generic_np_type) -> np.float64 | np.ndarray:
    return np.arccosh(array)


@nx.arctan.register
def _(array: generic_np_type) -> np.float64 | np.ndarray:
    return np.arctan(array)


@nx.arctanh.register
def _(array: generic_np_type) -> np.float64 | np.ndarray:
    return np.arctanh(array)


@nx.argmin.register
def _(array: generic_np_type) -> int:
    return np.argmin(array)


@nx.array.register
def _(array: generic_np_type) -> np.ndarray:
    return np.array(array)


@nx.block.register
def _(arrays: generic_np_type) -> np.ndarray:
    return np.block(arrays)


@nx.conjugate.register
def _(array: generic_np_type) -> generic_np_type:
    return np.conjugate(array)


@nx.cos.register
def _(array: generic_np_type) -> generic_np_type:
    return np.cos(array)


@nx.diag.register
def _(array: generic_list_type) -> np.ndarray:
    return np.diag(array)


@nx.diagonal.register
def _(array: generic_list_type) -> np.ndarray:
    return np.diagonal(array)


@nx.eye.register
def _(n: int) -> np.ndarray:
    return np.eye(n)


@nx.exp.register
def _(array: generic_np_type) -> np.float64 | np.ndarray:
    return np.exp(array)


@linalg.expm.register
def _(array: generic_np_type) -> np.float64 | np.ndarray:
    return scipy.linalg.expm(array)


@linalg.cholesky.register
def _(array: generic_list_type) -> np.ndarray:
    return np.linalg.cholesky(array)


@linalg.det.register
def _(array: generic_list_type) -> np.float64 | np.complex128:
    return np.linalg.det(array)


@linalg.eigh.register
def _(array: generic_list_type) -> tuple[np.ndarray, np.ndarray]:
    return np.linalg.eigh(array)


@linalg.inv.register
def _(array: generic_list_type) -> np.ndarray:
    return np.linalg.inv(array)


@linalg.logm.register
def _(array: generic_np_type) -> np.float64 | np.ndarray:
    return scipy.linalg.logm(array)


@linalg.norm.register
def _(array: generic_np_type) -> np.float64:
    return np.linalg.norm(array)


@linalg.qr.register
def _(array: generic_list_type) -> tuple[np.ndarray, np.ndarray]:
    return np.linalg.qr(array)


@linalg.solve.register
def _(array_a: generic_list_type, array_b: generic_list_type) -> np.ndarray:
    return np.linalg.solve(array_a, array_b)


@linalg.solve_continuous_lyapunov.register
def _(array_a: generic_list_type, array_q: generic_list_type) -> np.ndarray:
    return scipy.linalg.solve_continuous_lyapunov(array_a, array_q)


@linalg.svd.register
def _(array: generic_list_type) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.linalg.svd(array)


@nx.tensordot.register
def _(
    array_a: generic_np_type, array_b: generic_np_type, *, axes: int = 2
) -> np.float64 | np.ndarray:
    return np.tensordot(array_a, array_b, axes=axes)


@nx.tanh.register
def _(array: generic_np_type) -> np.float64 | np.ndarray:
    return np.tanh(array)
