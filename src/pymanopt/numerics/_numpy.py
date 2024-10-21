from numbers import Number
from typing import Any, Optional, Sequence, Tuple, Union, override

import numpy as np
import numpy.testing as np_testing
import packaging.version as pv
import scipy
import scipy.linalg

from pymanopt.numerics.array_t import array_t
from pymanopt.numerics.core import NumericsBackend


np_array_t = np.ndarray


class NumpyNumericsBackend(NumericsBackend):
    _dtype: np.dtype

    def __init__(self, dtype: np.dtype = np.float64):
        assert dtype in {np.float32, np.float64, np.complex64, np.complex128}
        self._dtype = dtype

    @property
    @override
    def dtype(self):
        return self._dtype

    @property
    @override
    def is_dtype_real(self):
        return np.issubdtype(self.dtype, np.floating)

    @override
    @staticmethod
    def DEFAULT_REAL_DTYPE():
        return np.array([1.0]).dtype

    @override
    @staticmethod
    def DEFAULT_COMPLEX_DTYPE():
        return np.array([1j]).dtype

    @override
    def __repr__(self):
        return f"NumpyNumericsBackend(dtype={self.dtype})"

    @override
    def to_real_backend(self) -> "NumpyNumericsBackend":
        if self.is_dtype_real:
            return self
        if self.dtype == np.complex64:
            return NumpyNumericsBackend(dtype=np.float32)
        elif self.dtype == np.complex128:
            return NumpyNumericsBackend(dtype=np.float64)
        else:
            raise ValueError(f"dtype {self.dtype} is not supported")

    @override
    def to_complex_backend(self) -> "NumpyNumericsBackend":
        if not self.is_dtype_real:
            return self
        if self.dtype == np.float32:
            return NumpyNumericsBackend(dtype=np.complex64)
        elif self.dtype == np.float64:
            return NumpyNumericsBackend(dtype=np.complex128)
        else:
            raise ValueError(f"dtype {self.dtype} is not supported")

    ##############################################################################
    # Numerics functions
    ##############################################################################

    @override
    def abs(self, array: np_array_t) -> np_array_t:
        return np.abs(array)

    @override
    def all(self, array: np_array_t) -> bool:
        return np.all(array).item()

    @override
    def allclose(
        self,
        array_a: np_array_t,
        array_b: np_array_t,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        return np.allclose(array_a, array_b, rtol, atol)

    @override
    def any(self, array: np_array_t) -> bool:
        return np.any(array).item()

    @override
    def arange(self, *args: int) -> np_array_t:
        return np.arange(*args)

    @override
    def arccos(self, array: np_array_t) -> np_array_t:
        return np.arccos(array)

    @override
    def arccosh(self, array: np_array_t) -> np_array_t:
        return np.arccosh(array)

    @override
    def arctan(self, array: np_array_t) -> np_array_t:
        return np.arctan(array)

    @override
    def arctanh(self, array: np_array_t) -> np_array_t:
        return np.arctanh(array)

    @override
    def argmin(self, array: np_array_t):
        return np.argmin(array)

    @override
    def argsort(self, array: np_array_t):
        return np.argsort(array)

    @override
    def array(self, array: array_t) -> np_array_t:  # type: ignore
        return np.asarray(array, dtype=self.dtype)

    @override
    def assert_allclose(
        self,
        array_a: np_array_t,
        array_b: np_array_t,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        np_testing.assert_allclose(
            array_a, array_b, rtol, atol, equal_nan=False
        )

    @override
    def assert_equal(
        self,
        array_a: np_array_t,
        array_b: np_array_t,
    ) -> None:
        return np_testing.assert_equal(array_a, array_b)

    @override
    def block(self, arrays: Sequence[np_array_t]) -> np_array_t:
        return np.block(arrays)

    @override
    def conjugate(self, array: np_array_t) -> np_array_t:
        return np.conjugate(array)

    @override
    def cos(self, array: np_array_t) -> np_array_t:
        return np.cos(array)

    @override
    def diag(self, array: np_array_t) -> np_array_t:
        return np.diag(array)

    @override
    def diagonal(
        self, array: np_array_t, axis1: int, axis2: int
    ) -> np_array_t:
        return np.diagonal(array, axis1, axis2)

    @override
    def eps(self) -> float:
        return np.finfo(self.dtype).eps

    @override
    def exp(self, array: np_array_t) -> np_array_t:
        return np.exp(array)

    @override
    def expand_dims(self, array: np_array_t, axis: int) -> np_array_t:
        return np.expand_dims(array, axis)

    @override
    def eye(self, size: int) -> np_array_t:
        return np.eye(size, dtype=self.dtype)

    @override
    def hstack(self, arrays: Sequence[np_array_t]) -> np_array_t:
        return np.hstack(arrays)

    @override
    def iscomplexobj(self, array: np_array_t) -> bool:
        return np.iscomplexobj(array)

    @override
    def isnan(self, array: np_array_t) -> np_array_t:
        return np.isnan(array)

    @override
    def isrealobj(self, array: np_array_t) -> bool:
        return np.isrealobj(array)

    @override
    def linalg_cholesky(self, array: np_array_t) -> np_array_t:
        return np.linalg.cholesky(array)

    @override
    def linalg_det(self, array: np_array_t) -> np_array_t:
        return np.linalg.det(array)

    @override
    def linalg_eigh(self, array: np_array_t) -> Tuple[np_array_t, np_array_t]:
        return np.linalg.eigh(array)

    @override
    def linalg_eigvalsh(
        self, array_x: np_array_t, array_y: Optional[np_array_t] = None
    ) -> np_array_t:
        if array_y is None:
            return np.linalg.eigvalsh(array_x)
        else:
            return np.vectorize(
                scipy.linalg.eigvalsh, signature="(m,m),(m,m)->(m)"
            )(array_x, array_y)

    @override
    def linalg_expm(
        self, array: np_array_t, symmetric: bool = False
    ) -> np_array_t:
        if not symmetric:
            # Scipy 1.9.0 added support for calling scipy.linalg.expm on stacked
            # matrices.
            if pv.parse(scipy.version.version) >= pv.parse("1.9.0"):
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

    @override
    def linalg_inv(self, array: np_array_t) -> np_array_t:
        return np.linalg.inv(array)

    @override
    def linalg_logm(
        self, array: np_array_t, positive_definite: bool = False
    ) -> np_array_t:
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

    @override
    def linalg_matrix_rank(self, array: np_array_t) -> int:
        return np.linalg.matrix_rank(array)

    @override
    def linalg_norm(
        self, array: np_array_t, *args: Any, **kwargs: Any
    ) -> np_array_t:
        return np.linalg.norm(array, *args, **kwargs)  # type: ignore

    @override
    def linalg_qr(self, array: np_array_t) -> Tuple[np_array_t, np_array_t]:
        q, r = np.linalg.qr(array)

        return q, r
        # Compute signs or unit-modulus phase of entries of diagonal of r.
        s = np.diagonal(r, axis1=-2, axis2=-1).copy()
        s[s == 0] = 1
        s = s / np.abs(s)
        s = np.expand_dims(s, axis=-1)
        # normalize q and r to have either 1 or unit-modulus on the diagonal of r
        q = q * self.transpose(s)
        r = r * np.conjugate(s)
        return q, r

    @override
    def linalg_solve(
        self, array_a: np_array_t, array_b: np_array_t
    ) -> np_array_t:
        return np.linalg.solve(array_a, array_b)

    @override
    def linalg_solve_continuous_lyapunov(
        self, array_a: np_array_t, array_q: np_array_t
    ) -> np_array_t:
        return scipy.linalg.solve_continuous_lyapunov(array_a, array_q)

    @override
    def linalg_svd(
        self, array: np_array_t, *args: Any, **kwargs: Any
    ) -> Tuple[np_array_t, np_array_t, np_array_t]:
        return np.linalg.svd(array, *args, **kwargs)

    @override
    def log(self, array: np_array_t) -> np_array_t:
        return np.log(array)

    @override
    def logspace(self, *args: int) -> np_array_t:
        return np.logspace(*args, dtype=self.dtype)

    @override
    def ndim(self, array: np_array_t) -> int:
        return array.ndim

    @override
    def ones(self, shape: Sequence[int]) -> np_array_t:
        return np.ones(shape, self.dtype)

    @override
    def prod(self, array: np_array_t) -> float:
        return np.prod(array)  # type: ignore

    @override
    def random_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Union[int, Sequence[int], None] = None,
    ) -> np_array_t:
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

    @override
    def random_uniform(
        self, size: Union[int, Sequence[int], None] = None
    ) -> np_array_t:
        if self.is_dtype_real:
            return np.asarray(np.random.uniform(size=size), dtype=self.dtype)
        else:
            real_dtype = np.finfo(self.dtype).dtype
            return np.asarray(
                np.random.uniform(size=size), dtype=real_dtype
            ) + 1j * np.asarray(np.random.uniform(size=size), dtype=real_dtype)

    @override
    def real(self, array: np_array_t) -> np_array_t:
        return np.real(array)

    @override
    def reshape(
        self, array: np.ndarray, newshape: Sequence[int]
    ) -> np.ndarray:
        return np.reshape(array, newshape)

    @override
    def sin(self, array: np_array_t) -> np_array_t:
        return np.sin(array)

    @override
    def sinc(self, array: np_array_t) -> np_array_t:
        return np.sinc(array)

    @override
    def sort(self, array: np_array_t) -> np_array_t:
        return np.sort(array)

    @override
    def spacing(self, array: np_array_t) -> np_array_t:
        return np.spacing(array)  # type: ignore

    @override
    def sqrt(self, array: np_array_t) -> np_array_t:
        return np.sqrt(array)

    @override
    def squeeze(self, array: np_array_t) -> np_array_t:
        return np.squeeze(array)

    @override
    def stack(self, arrays: Sequence[np_array_t], axis: int = 0) -> np_array_t:
        return np.stack(arrays)

    @override
    def sum(self, array: np_array_t, *args: Any, **kwargs: Any) -> np_array_t:
        return np.sum(array, *args, **kwargs)  # type: ignore

    @override
    def tan(self, array: np_array_t) -> np_array_t:
        return np.tan(array)

    @override
    def tanh(self, array: np_array_t) -> np_array_t:
        return np.tanh(array)

    @override
    def tensordot(
        self, a: np_array_t, b: np_array_t, axes: int = 2
    ) -> np_array_t:
        return np.tensordot(a, b, axes=axes)

    @override
    def tile(self, array: np_array_t, reps: int | Sequence[int]) -> np_array_t:
        return np.tile(array, reps)

    @override
    def trace(self, array: np_array_t) -> Union[np_array_t, Number]:
        return (
            np.trace(array).item()
            if array.ndim == 2
            else np.trace(array, axis1=-2, axis2=-1)
        )

    @override
    def transpose(self, array: np_array_t) -> np_array_t:
        new_shape = list(range(self.ndim(array)))
        new_shape[-1], new_shape[-2] = new_shape[-2], new_shape[-1]
        return np.transpose(array, new_shape)

    @override
    def triu_indices(self, n: int, k: int = 0) -> np_array_t:
        return np.triu_indices(n, k)

    @override
    def vstack(self, arrays: Sequence[np_array_t]) -> np_array_t:
        return np.vstack(arrays)

    @override
    def where(
        self,
        condition: np_array_t,
        x: Optional[np_array_t] = None,
        y: Optional[np_array_t] = None,
    ):
        if x is None and y is None:
            return np.where(condition)
        elif x is not None and y is not None:
            return np.where(condition, x, y)
        else:
            raise ValueError(
                f"Both x and y have to be specified but are respectively {x} and {y}"
            )

    @override
    def zeros(self, shape: Sequence[int]) -> np_array_t:
        return np.zeros(shape, dtype=self.dtype)

    @override
    def zeros_like(self, array: np_array_t) -> np_array_t:
        return np.zeros_like(array, dtype=self.dtype)
