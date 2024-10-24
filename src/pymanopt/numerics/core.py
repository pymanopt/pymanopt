from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Optional, TypeVar, Union

import numpy as np
import scipy.special

from pymanopt.numerics.array_t import array_t


__all__ = ["NumericsBackend", "TupleOrList"]


def not_implemented(function):
    @wraps(function)
    def inner(self, *args, **kwargs):
        raise NotImplementedError(
            f"Function '{function.__name__}' not implemented for backend {self}"
        )

    return inner


T = TypeVar("T")
TupleOrList = Union[list[T], tuple[T, ...]]


class NumericsBackend(ABC):
    @property
    @abstractmethod
    def dtype(self):
        pass

    @property
    @abstractmethod
    def is_dtype_real(self):
        pass

    @staticmethod
    @abstractmethod
    def DEFAULT_REAL_DTYPE():
        pass

    @staticmethod
    @abstractmethod
    def DEFAULT_COMPLEX_DTYPE():
        pass

    @abstractmethod
    def __repr__(self):
        pass

    def __eq__(self, other):
        return repr(self) == repr(other)

    @abstractmethod
    def to_complex_backend(self) -> "NumericsBackend":
        pass

    @abstractmethod
    def to_real_backend(self) -> "NumericsBackend":
        pass

    ##############################################################################
    # Numerics functions
    ##############################################################################

    @not_implemented
    def abs(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def all(self, array: array_t) -> Optional[bool]:  # type: ignore
        pass

    @not_implemented
    def allclose(
        self, array_a: array_t, array_b: array_t, rtol: float, atol: float
    ) -> Optional[bool]:  # type:ignore
        pass

    @not_implemented
    def any(self, array: array_t) -> Optional[bool]:  # type: ignore
        pass

    @not_implemented
    def arange(self, start: int, stop: int, step: int) -> array_t:  # type: ignore
        pass

    @not_implemented
    def arccos(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def arccosh(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def arctan(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def arctanh(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def argmin(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def argsort(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def array(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def assert_allclose(
        self,
        array_a: array_t,  # type: ignore
        array_b: array_t,  # type: ignore
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        pass

    def assert_almost_equal(
        self,
        array_a: array_t,  # type: ignore
        array_b: array_t,  # type: ignore
    ) -> None:
        self.assert_allclose(array_a, array_b)

    def assert_array_almost_equal(
        self,
        array_a: array_t,  # type: ignore
        array_b: array_t,  # type: ignore
    ) -> None:
        self.assert_allclose(array_a, array_b)

    @not_implemented
    def assert_equal(
        self,
        array_a: array_t,  # type: ignore
        array_b: array_t,  # type: ignore
    ) -> None:
        pass

    @not_implemented
    def block(self, arrays: TupleOrList[array_t]) -> array_t:  # type: ignore
        pass

    @not_implemented
    def concatenate(
        self,
        arrays: TupleOrList[array_t],  # type: ignore
        axis: int = 0,
    ) -> array_t:  # type: ignore
        pass

    @not_implemented
    def conjugate(self, array: array_t) -> array_t:  # type: ignore
        pass

    def conjugate_transpose(self, array: array_t) -> array_t:  # type: ignore
        return self.conjugate(self.transpose(array))

    @not_implemented
    def cos(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def diag(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def diagonal(self, array: array_t, axis1: int, axis2: int) -> array_t:
        pass

    @not_implemented
    def eps(self, dtype) -> float:  # type: ignore
        pass

    @not_implemented
    def exp(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def expand_dims(self, array: array_t, axis: int) -> array_t:  # type: ignore
        pass

    @not_implemented
    def eye(self, size: int) -> array_t:  # type: ignore
        pass

    @not_implemented
    def hstack(self, arrays: TupleOrList[array_t]) -> array_t:  # type: ignore
        pass

    def herm(self, array: array_t) -> array_t:  # type: ignore
        return 0.5 * (array + self.conjugate_transpose(array))

    @not_implemented
    def iscomplexobj(self, array: array_t) -> bool:  # type: ignore
        pass

    @not_implemented
    def isnan(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def isrealobj(self, array: array_t) -> bool:  # type: ignore
        pass

    @not_implemented
    def linalg_cholesky(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def linalg_det(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def linalg_eigh(self, array: array_t) -> tuple[array_t, array_t]:  # type: ignore
        pass

    @not_implemented
    def linalg_eigvalsh(
        self,
        array_x: array_t,  # type: ignore
        array_y: Optional[array_t] = None,  # type: ignore
    ) -> array_t:  # type: ignore
        pass

    @not_implemented
    def linalg_expm(
        self,
        array: array_t,  # type: ignore
        symmetric: bool = False,
    ) -> array_t:  # type: ignore
        pass

    @not_implemented
    def linalg_inv(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def linalg_matrix_rank(self, array: array_t) -> int:  # type: ignore
        pass

    @not_implemented
    def linalg_logm(
        self, array: array_t, positive_definite: bool = False
    ) -> array_t:
        pass

    @not_implemented
    def linalg_norm(self, array: array_t, *args, **kwargs) -> array_t:  # type: ignore
        pass

    @not_implemented
    def linalg_qr(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def linalg_solve(self, array_a: array_t, array_b: array_t) -> array_t:
        pass

    @not_implemented
    def linalg_solve_continuous_lyapunov(
        self,
        array_a: array_t,  # type: ignore
        array_q: array_t,  # type: ignore
    ) -> array_t:  # type: ignore
        pass

    @not_implemented
    def linalg_svd(
        self,
        array: array_t,  # type: ignore
        *args,
        **kwargs,
    ) -> tuple[array_t, array_t, array_t]:  # type: ignore
        pass

    @not_implemented
    def log(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def logspace(self, *args: int) -> array_t:  # type: ignore
        pass

    def matvec(self, A: array_t, x: array_t) -> array_t:  # type: ignore
        pass

    def matmul(self, A: array_t, B: array_t) -> array_t:  # type: ignore
        pass

    def multieye(self, k: int, n: int) -> array_t:  # type: ignore
        return self.tile(self.eye(n), (k, 1, 1))

    @not_implemented
    def ndim(self, array: array_t) -> int:  # type: ignore
        pass

    newaxis = None

    @not_implemented
    def ones(self, shape: TupleOrList[int]) -> array_t:  # type: ignore
        pass

    @not_implemented
    def ones_bool(self, shape: TupleOrList[int]) -> array_t:  # type: ignore
        pass

    pi = np.pi

    #   - np.polyfit
    #   - np.polyval

    @not_implemented
    def prod(self, array: array_t) -> float:  # type: ignore
        pass

    @not_implemented
    def random_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Union[int, TupleOrList[int], None] = None,
    ) -> array_t:  # type: ignore
        pass

    def random_randn(self, *dims: int) -> array_t:  # type: ignore
        return self.random_normal(loc=0.0, scale=1.0, size=dims)

    @not_implemented
    def random_uniform(
        self, size: Union[int, TupleOrList[int], None] = None
    ) -> array_t:  # type: ignore
        pass

    @not_implemented
    def real(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def reshape(
        self,
        array: array_t,  # type: ignore
        newshape: TupleOrList[int],
    ) -> array_t:  # type: ignore
        pass

    # TODO: seterr
    # def seterr(all=None):
    #     np.seterr(all=all)
    #     if all == 'raise':
    #         try:
    #             import torch
    #             torch.autograd.set_detect_anomaly(True)
    #         except ImportError:
    #             pass
    #         try:
    #             import jax
    #             jax.config.update("jax_debug_nans", True)
    #         except ImportError:
    #             pass
    #         try:
    #             import tensorflow as tf
    #             tf.debugging.enable_check_numerics()
    #         except ImportError:
    #             pass

    @not_implemented
    def sin(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def sinc(self, array: array_t) -> array_t:  # type: ignore
        pass

    def skew(self, array: array_t) -> array_t:  # type: ignore
        return 0.5 * (array - self.transpose(array))

    def skewh(self, array: array_t) -> array_t:  # type: ignore
        return 0.5 * (array - self.conjugate_transpose(array))

    @not_implemented
    def sort(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def spacing(self, array: array_t) -> array_t:  # type: ignore
        pass

    def special_comb(self, n: int, k: int):
        return scipy.special.comb(n, k)

    @not_implemented
    def sqrt(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def squeeze(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def stack(
        self,
        arrays: TupleOrList[array_t],  # type: ignore
        axis: int = 0,
    ) -> array_t:  # type: ignore
        pass

    @not_implemented
    def sum(
        self,
        array: array_t,  # type: ignore
        *args: Any,
        **kwargs: Any,
    ) -> array_t:  # type: ignore
        pass

    def sym(self, array: array_t) -> array_t:  # type: ignore
        return 0.5 * (array + self.transpose(array))

    @not_implemented
    def tan(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def tanh(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def tensordot(self, a: array_t, b: array_t, axes: int) -> array_t:  # type: ignore
        pass

    @not_implemented
    def tile(self, array: array_t, reps: int | TupleOrList[int]) -> array_t:
        pass

    @not_implemented
    def trace(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def transpose(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def triu(self, array: array_t, k: int = 0) -> array_t:  # type:ignore
        pass

    @not_implemented
    def vstack(self, arrays: TupleOrList[array_t]) -> array_t:  # type: ignore
        pass

    @not_implemented
    def where(
        self,
        condition: array_t,  # type: ignore
        x: Optional[array_t],  # type: ignore
        y: Optional[array_t],  # type: ignore
    ) -> Union[array_t, tuple[array_t, ...]]:  # type: ignore
        pass

    @not_implemented
    def zeros(self, shape: TupleOrList[int]) -> array_t:  # type: ignore
        pass

    @not_implemented
    def zeros_bool(self, shape: TupleOrList[int]) -> array_t:  # type: ignore
        pass

    @not_implemented
    def zeros_like(self, array: array_t) -> array_t:  # type: ignore
        pass
