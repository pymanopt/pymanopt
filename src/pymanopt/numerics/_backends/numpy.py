import numpy as np
import scipy
from typing import Sequence

import pymanopt.numerics as nx

array_like = int | float | complex | np.generic | tuple | list | np.ndarray


@nx.abs.register
def _(array: array_like) -> np.float64 | np.ndarray:
    return np.abs(array)


@nx.all.register
def _(array: array_like) -> bool:
    return np.all(array)


@nx.allclose.register
def _(array_a: array_like, array_b: array_like) -> bool:
    return np.allclose(array_a, array_b)


@nx.any.register
def _(array: array_like) -> bool:
    return np.any(array)


@nx.arccos.register
def _(array: array_like) -> np.float64 | np.ndarray:
    return np.arccos(array)


@nx.arccosh.register
def _(array: array_like) -> np.float64 | np.ndarray:
    return np.arccosh(array)


@nx.arctan.register
def _(array: array_like) -> np.float64 | np.ndarray:
    return np.arctan(array)


@nx.arctanh.register
def _(array: array_like) -> np.float64 | np.ndarray:
    return np.arctanh(array)


@nx.argmin.register
def _(array: array_like) -> int:
    return np.argmin(array)


@nx.array.register
def _(array: array_like) -> np.ndarray:
    return np.array(array)


@nx.block.register
def _(arrays: Sequence[np.ndarray]) -> np.ndarray:
    return np.block(arrays)


@nx.conjugate.register
def _(array: array_like) -> array_like:
    return np.conjugate(array)


@nx.cos.register
def _(array: array_like) -> array_like:
    return np.cos(array)


@nx.diag.register
def _(array: array_like) -> np.ndarray:
    return np.diag(array)


@nx.diagonal.register
def _(array: array_like) -> np.ndarray:
    return np.diagonal(array)


@nx.exp.register
def _(array: array_like) -> np.float64 | np.ndarray:
    return np.exp(array)


@nx.expand_dims.register
def _(
    array: array_like,
    axis: int | tuple[int, ...] = None
) -> np.ndarray:
    return np.expand_dims(array, axis)


@nx.iscomplexobj.register
def _(array: array_like) -> bool:
    return np.iscomplexobj(array)


@nx.isnan.register
def _(array: array_like) -> bool:
    return np.isnan(array)


@nx.isrealobj.register
def _(array: array_like) -> bool:
    return np.isrealobj(array)


@nx.log.register
def _(array: array_like) -> np.float64 | np.ndarray:
    return np.log(array)


@nx.linalg.cholesky.register
def _(array: array_like) -> np.ndarray:
    return np.linalg.cholesky(array)


@nx.linalg.det.register
def _(array: array_like) -> np.float64 | np.complex128:
    return np.linalg.det(array)


@nx.linalg.eigh.register
def _(array: array_like) -> tuple[np.ndarray, np.ndarray]:
    return np.linalg.eigh(array)


@nx.linalg.expm.register
def _(array: array_like) -> np.float64 | np.ndarray:
    return scipy.linalg.expm(array)


@nx.linalg.inv.register
def _(array: array_like) -> np.ndarray:
    return np.linalg.inv(array)


@nx.linalg.logm.register
def _(array: array_like) -> np.float64 | np.ndarray:
    return scipy.linalg.logm(array)


@nx.linalg.matrix_rank.register
def _(array: array_like) -> int:
    return np.linalg.matrix_rank(array)


@nx.linalg.norm.register
def _(
    array: array_like,
    *args: tuple,
    **kwargs: dict,
) -> np.float64:
    return np.linalg.norm(array, *args, **kwargs)


@nx.linalg.qr.register
def _(array: array_like) -> tuple[np.ndarray, np.ndarray]:
    return np.linalg.qr(array)


@nx.linalg.solve.register
def _(array_a: array_like, array_b: array_like) -> np.ndarray:
    return np.linalg.solve(array_a, array_b)


@nx.linalg.solve_continuous_lyapunov.register
def _(array_a: array_like, array_q: array_like) -> np.ndarray:
    return scipy.linalg.solve_continuous_lyapunov(array_a, array_q)


@nx.linalg.svd.register
def _(
    array: array_like,
    *args: tuple,
    **kwargs: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.linalg.svd(array, *args, **kwargs)


@nx.polyfit.register
def _(x: array_like, y: array_like, deg: int) -> np.ndarray:
    return np.polyfit(x, y, deg)


@nx.polyval.register
def _(p: array_like, x: array_like) -> np.ndarray:
    return np.polyval(p, x)


@nx.prod.register
def _(array: array_like) -> np.float64:
    return np.prod(array)


@nx.real.register
def _(array: array_like) -> int | float | np.ndarray:
    return np.real(array)


@nx.sin.register
def _(array: array_like) -> np.float64 | np.ndarray:
    return np.sin(array)


@nx.sinc.register
def _(array: array_like) -> np.float64 | np.ndarray:
    return np.sinc(array)


@nx.sort.register
def _(array: array_like) -> np.ndarray:
    return np.sort(array)


@nx.spacing.register
def _(array: array_like) -> np.float64 | np.ndarray:
    return np.spacing(array)


@nx.sqrt.register
def _(array: array_like) -> np.float64 | np.ndarray:
    return np.sqrt(array)


@nx.sum.register
def _(
    array: array_like,
    *args: tuple,
    **kwargs: dict
) -> np.float64 | np.ndarray:
    return np.sum(array, *args, **kwargs)


@nx.tanh.register
def _(array: array_like) -> np.float64 | np.ndarray:
    return np.tanh(array)


@nx.tensordot.register
def _(
    array_a: array_like,
    array_b: array_like,
    *args: tuple,
    **kwargs: dict
) -> np.float64 | np.ndarray:
    return np.tensordot(array_a, array_b, *args, **kwargs)


@nx.tile.register
def _(array: array_like, reps: int | tuple[int, ...]) -> np.ndarray:
    return np.tile(array, reps)


@nx.trace.register
def _(
    array: array_like,
    *args: tuple,
    **kwargs: dict
) -> np.float64:
    return np.trace(array, *args, **kwargs)


@nx.transpose.register
def _(array: array_like, axes: tuple[int, ...] | None = None) -> np.ndarray:
    return np.transpose(array, axes)


@nx.where.register
def _(condition: array_like) -> np.ndarray:
    return np.where(condition)


@nx.zeros_like.register
def _(array: array_like) -> np.ndarray:
    return np.zeros_like(array)
