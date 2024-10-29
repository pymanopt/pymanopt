from numbers import Number
from typing import Any, Literal, Optional, Union

import numpy as np
import numpy.testing as np_testing
import packaging.version as pv
import scipy
import scipy.linalg

from pymanopt.backends.backend import Backend, TupleOrList


def _raise_not_implemented_error(*args, **kwargs):
    raise NotImplementedError(
        "No autodiff support available for the NumPy backend"
    )


class NumpyBackend(Backend):
    ##########################################################################
    # Common attributes, properties and methods
    ##########################################################################
    array_t = np.ndarray

    def __init__(self, dtype: type = np.float64):
        assert (
            dtype == np.float32
            or dtype == np.float64
            or dtype == np.complex64
            or dtype == np.complex128
        ), f"dtype {dtype} is not supported"
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_dtype_real(self):
        return np.issubdtype(self.dtype, np.floating)

    @staticmethod
    def DEFAULT_REAL_DTYPE():
        return np.array([1.0]).dtype

    @staticmethod
    def DEFAULT_COMPLEX_DTYPE():
        return np.array([1j]).dtype

    def __repr__(self):
        return f"NumpyBackend(dtype={self.dtype})"

    def to_real_backend(self) -> "NumpyBackend":
        if self.is_dtype_real:
            return self
        if self.dtype == np.complex64:
            return NumpyBackend(dtype=np.float32)
        elif self.dtype == np.complex128:
            return NumpyBackend(dtype=np.float64)
        else:
            raise ValueError(f"dtype {self.dtype} is not supported")

    def to_complex_backend(self) -> "NumpyBackend":
        if not self.is_dtype_real:
            return self
        if self.dtype == np.float32:
            return NumpyBackend(dtype=np.complex64)
        elif self.dtype == np.float64:
            return NumpyBackend(dtype=np.complex128)
        else:
            raise ValueError(f"dtype {self.dtype} is not supported")

    ##############################################################################
    # Autodiff methods
    ##############################################################################

    def prepare_function(self, function):
        return function

    generate_gradient_operator = _raise_not_implemented_error
    generate_hessian_operator = _raise_not_implemented_error

    ##############################################################################
    # Numerics functions
    ##############################################################################

    def abs(self, array: np.ndarray) -> np.ndarray:
        return np.abs(array)

    def all(self, array: np.ndarray) -> bool:
        return np.all(array).item()

    def allclose(
        self,
        array_a: np.ndarray,
        array_b: np.ndarray,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        return np.allclose(array_a, array_b, rtol, atol)

    def any(self, array: np.ndarray) -> bool:
        return np.any(array).item()

    def arange(
        self,
        start: int,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> np.ndarray:
        return np.arange(start, stop, step)

    def arccos(self, array: np.ndarray) -> np.ndarray:
        return np.arccos(array)

    def arccosh(self, array: np.ndarray) -> np.ndarray:
        return np.arccosh(array)

    def arctan(self, array: np.ndarray) -> np.ndarray:
        return np.arctan(array)

    def arctanh(self, array: np.ndarray) -> np.ndarray:
        return np.arctanh(array)

    def argmin(self, array: np.ndarray):
        return np.argmin(array)

    def argsort(self, array: np.ndarray):
        return np.argsort(array)

    def array(self, array: Any) -> np.ndarray:  # type: ignore
        return np.asarray(array, dtype=self.dtype)

    def assert_allclose(
        self,
        array_a: np.ndarray,
        array_b: np.ndarray,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        np_testing.assert_allclose(
            array_a, array_b, rtol, atol, equal_nan=False
        )

    def assert_equal(
        self,
        array_a: np.ndarray,
        array_b: np.ndarray,
    ) -> None:
        return np_testing.assert_equal(array_a, array_b)

    def concatenate(
        self, arrays: TupleOrList[np.ndarray], axis: int = 0
    ) -> np.ndarray:
        return np.concatenate(arrays, axis)

    def conjugate(self, array: np.ndarray) -> np.ndarray:
        return np.conjugate(array)

    def cos(self, array: np.ndarray) -> np.ndarray:
        return np.cos(array)

    def diag(self, array: np.ndarray) -> np.ndarray:
        return np.diag(array)

    def diagonal(
        self, array: np.ndarray, axis1: int, axis2: int
    ) -> np.ndarray:
        return np.diagonal(array, axis1, axis2)

    def eps(self) -> float:
        return float(np.finfo(self.dtype).eps)

    def exp(self, array: np.ndarray) -> np.ndarray:
        return np.exp(array)

    def expand_dims(self, array: np.ndarray, axis: int) -> np.ndarray:
        return np.expand_dims(array, axis)

    def eye(self, size: int) -> np.ndarray:
        return np.eye(size, dtype=self.dtype)

    def hstack(self, arrays: TupleOrList[np.ndarray]) -> np.ndarray:
        return np.hstack(arrays)

    def iscomplexobj(self, array: np.ndarray) -> bool:
        return np.iscomplexobj(array)

    def isnan(self, array: np.ndarray) -> np.ndarray:
        return np.isnan(array)

    def isrealobj(self, array: np.ndarray) -> bool:
        return np.isrealobj(array)

    def linalg_cholesky(self, array: np.ndarray) -> np.ndarray:
        return np.linalg.cholesky(array)

    def linalg_det(self, array: np.ndarray) -> np.ndarray:
        return np.linalg.det(array)

    def linalg_eigh(self, array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.linalg.eigh(array)

    def linalg_eigvalsh(
        self, array_x: np.ndarray, array_y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if array_y is None:
            return np.linalg.eigvalsh(array_x)
        else:
            return np.vectorize(
                scipy.linalg.eigvalsh, signature="(m,m),(m,m)->(m)"
            )(array_x, array_y)

    def linalg_expm(
        self, array: np.ndarray, symmetric: bool = False
    ) -> np.ndarray:
        if not symmetric:
            # Scipy 1.9.0 added support for calling scipy.linalg.expm on stacked
            # matrices.
            if pv.parse(scipy.__version__) >= pv.parse("1.9.0"):
                scipy_expm = scipy.linalg.expm
            else:
                scipy_expm = np.vectorize(
                    scipy.linalg.expm, signature="(m,m)->(m,m)"
                )
            return scipy_expm(array)

        w, v = np.linalg.eigh(array)
        w = np.expand_dims(np.exp(w), axis=-1)
        expmA = v @ (w * self.conjugate_transpose(v))
        if np.isrealobj(array):
            return np.real(expmA)
        return expmA

    def linalg_inv(self, array: np.ndarray) -> np.ndarray:
        return np.linalg.inv(array)

    def linalg_logm(
        self, array: np.ndarray, positive_definite: bool = False
    ) -> np.ndarray:
        if not positive_definite:
            return np.vectorize(scipy.linalg.logm, signature="(m,m)->(m,m)")(
                array
            )

        w, v = np.linalg.eigh(array)
        w = np.expand_dims(np.log(w), axis=-1)
        logmA = v @ (w * self.conjugate_transpose(v))
        if np.isrealobj(array):
            return np.real(logmA)
        return logmA

    def linalg_matrix_rank(self, array: np.ndarray) -> int:
        return np.linalg.matrix_rank(array)

    def linalg_norm(
        self,
        array: np.ndarray,
        ord: Union[int, Literal["fro"], None] = None,
        axis: Union[int, TupleOrList[int], None] = None,
        keepdims: bool = False,
    ) -> Union[np.ndarray, Number]:
        return np.linalg.norm(array, ord=ord, axis=axis, keepdims=keepdims)

    def linalg_qr(self, array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q, r = np.linalg.qr(array)
        # Compute signs or unit-modulus phase of entries of diagonal of r.
        s = np.diagonal(r, axis1=-2, axis2=-1).copy()
        s[s == 0] = 1
        s = s / np.abs(s)
        s = np.expand_dims(s, axis=-1)
        # normalize q and r to have either 1 or unit-modulus on the diagonal of r
        q = q * self.transpose(s)
        r = r * np.conjugate(s)
        return q, r

    def linalg_solve(
        self, array_a: np.ndarray, array_b: np.ndarray
    ) -> np.ndarray:
        return np.linalg.solve(array_a, array_b)

    def linalg_solve_continuous_lyapunov(
        self, array_a: np.ndarray, array_q: np.ndarray
    ) -> np.ndarray:
        return scipy.linalg.solve_continuous_lyapunov(array_a, array_q)

    def linalg_svd(
        self,
        array: np.ndarray,
        full_matrices: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.linalg.svd(array, full_matrices=full_matrices)

    def linalg_svdvals(self, array: np.ndarray) -> np.ndarray:
        return np.linalg.svd(array, compute_uv=False)

    def log(self, array: np.ndarray) -> np.ndarray:
        return np.log(array)

    def log10(self, array: np.ndarray) -> np.ndarray:
        return np.log10(array)

    def logical_not(self, array: np.ndarray) -> np.ndarray:
        return np.logical_not(array)

    def logspace(self, start: float, stop: float, num: int) -> np.ndarray:
        return np.logspace(start, stop, num, dtype=self.dtype)

    def ndim(self, array: np.ndarray) -> int:
        return array.ndim

    def ones(self, shape: TupleOrList[int]) -> np.ndarray:
        return np.ones(shape, self.dtype)

    def ones_bool(self, shape: TupleOrList[int]) -> np.ndarray:
        return np.ones(shape, bool)

    def polyfit(
        self, x: np.ndarray, y: np.ndarray, deg: int = 1, full: bool = False
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        return np.polyfit(x, y, deg, full=full)  # type: ignore

    def polyval(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.polyval(p, x)

    def prod(self, array: np.ndarray) -> float:
        return np.prod(array)  # type: ignore

    def random_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Union[int, TupleOrList[int], None] = None,
    ) -> np.ndarray:
        if self.is_dtype_real:
            return np.asarray(
                np.random.normal(loc=loc, scale=scale, size=size),
                dtype=self.dtype,
            )
        else:
            real_dtype = np.finfo(self.dtype).dtype
            return np.asarray(
                np.random.normal(loc=loc, scale=scale, size=size),
                dtype=real_dtype,
            ) + 1j * np.asarray(
                np.random.normal(loc=loc, scale=scale, size=size),
                dtype=real_dtype,
            )

    def random_uniform(
        self, size: Union[int, TupleOrList[int], None] = None
    ) -> np.ndarray:
        if self.is_dtype_real:
            return np.asarray(np.random.uniform(size=size), dtype=self.dtype)
        else:
            real_dtype = np.finfo(self.dtype).dtype
            return np.asarray(
                np.random.uniform(size=size), dtype=real_dtype
            ) + 1j * np.asarray(np.random.uniform(size=size), dtype=real_dtype)

    def real(self, array: np.ndarray) -> np.ndarray:
        return np.real(array)

    def reshape(
        self, array: np.ndarray, newshape: TupleOrList[int]
    ) -> np.ndarray:
        return np.reshape(array, newshape)

    def sin(self, array: np.ndarray) -> np.ndarray:
        return np.sin(array)

    def sinc(self, array: np.ndarray) -> np.ndarray:
        return np.sinc(array)

    def sort(self, array: np.ndarray, descending: bool = False) -> np.ndarray:
        return np.sort(array)

    def spacing(self, array: np.ndarray) -> np.ndarray:
        return np.spacing(array)  # type: ignore

    def sqrt(self, array: np.ndarray) -> np.ndarray:
        return np.sqrt(array)

    def squeeze(self, array: np.ndarray) -> np.ndarray:
        return np.squeeze(array)

    def stack(
        self, arrays: TupleOrList[np.ndarray], axis: int = 0
    ) -> np.ndarray:
        return np.stack(arrays, axis=axis)

    def sum(
        self,
        array: np.ndarray,
        axis: Union[int, TupleOrList[int], None] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        return np.sum(array, axis=axis, keepdims=keepdims)  # type: ignore

    def tan(self, array: np.ndarray) -> np.ndarray:
        return np.tan(array)

    def tanh(self, array: np.ndarray) -> np.ndarray:
        return np.tanh(array)

    def tensordot(
        self, a: np.ndarray, b: np.ndarray, axes: int = 2
    ) -> np.ndarray:
        return np.tensordot(a, b, axes=axes)

    def tile(
        self, array: np.ndarray, reps: Union[int, TupleOrList[int]]
    ) -> np.ndarray:
        return np.tile(array, reps)

    def trace(self, array: np.ndarray) -> Union[np.ndarray, Number]:
        return (
            np.trace(array).item()
            if array.ndim == 2
            else np.trace(array, axis1=-2, axis2=-1)
        )

    def transpose(self, array: np.ndarray) -> np.ndarray:
        new_shape = list(range(self.ndim(array)))
        new_shape[-1], new_shape[-2] = new_shape[-2], new_shape[-1]
        return np.transpose(array, new_shape)

    def triu(self, array: np.ndarray, k: int = 0) -> np.ndarray:
        return np.triu(array, k)

    def vstack(self, arrays: TupleOrList[np.ndarray]) -> np.ndarray:
        return np.vstack(arrays)

    def where(
        self,
        condition: np.ndarray,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, tuple[np.ndarray, ...]]:
        if x is None and y is None:
            return np.where(condition)
        elif x is not None and y is not None:
            return np.where(condition, x, y)
        else:
            raise ValueError(
                f"Both x and y have to be specified but are respectively {x} and {y}"
            )

    def zeros(self, shape: TupleOrList[int]) -> np.ndarray:
        return np.zeros(shape, dtype=self.dtype)

    def zeros_bool(self, shape: TupleOrList[int]) -> np.ndarray:
        return np.zeros(shape, bool)

    def zeros_like(self, array: np.ndarray) -> np.ndarray:
        return np.zeros_like(array, dtype=self.dtype)
