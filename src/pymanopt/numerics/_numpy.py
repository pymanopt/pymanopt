from typing import Any, Optional, Sequence, Tuple

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

    def __init__(self, dtype=np.float64):
        assert dtype in {np.float32, np.float64, np.complex64, np.complex128}
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    def is_dtype_real(self):
        return np.issubdtype(self.dtype, np.floating)

    @property
    def DEFAULT_REAL_DTYPE(self):
        return np.array([1.0]).dtype

    @property
    def DEFAULT_COMPLEX_DTYPE(self):
        return np.array([1j]).dtype

    def __repr__(self):
        return f"NumpyNumericsBackend(dtype={self.dtype})"

    ##############################################################################
    # Numerics functions
    ##############################################################################

    def abs(self, array: np_array_t) -> np_array_t:
        return np.abs(array)

    def all(self, array: np_array_t) -> bool:
        return np.all(array).item()

    def allclose(self, array_a: np_array_t, array_b: np_array_t) -> bool:
        return np.allclose(array_a, array_b)

    def any(self, array: np_array_t) -> bool:
        return np.any(array).item()

    def arange(self, *args: int) -> np_array_t:
        return np.arange(*args)

    def arccos(self, array: np_array_t) -> np_array_t:
        return np.arccos(array)

    def arccosh(self, array: np_array_t) -> np_array_t:
        return np.arccosh(array)

    def arctan(self, array: np_array_t) -> np_array_t:
        return np.arctan(array)

    def arctanh(self, array: np_array_t) -> np_array_t:
        return np.arctanh(array)

    def argmin(self, array: np_array_t):
        return np.argmin(array)

    def argsort(self, array: np_array_t):
        return np.argsort(array)

    def array(self, array: array_t) -> np_array_t:  # type: ignore
        return (
            array
            if isinstance(array, np.ndarray) and array.dtype == self.dtype
            else np.asarray(array, dtype=self.dtype)
        )

    def assert_allclose(
        self,
        array_a: np_array_t,
        array_b: np_array_t,
        rtol: float = 1e-7,
        atol: float = 1e-10,
    ) -> None:
        return np_testing.assert_allclose(
            array_a, array_b, rtol=rtol, atol=atol
        )

    def assert_almost_equal(
        self, array_a: np_array_t, array_b: np_array_t
    ) -> None:
        return np_testing.assert_almost_equal(array_a, array_b)

    def assert_array_almost_equal(
        self, array_a: np_array_t, array_b: np_array_t
    ) -> None:
        return np_testing.assert_array_almost_equal(array_a, array_b)

    def assert_equal(
        self,
        array_a: np_array_t,
        array_b: np_array_t,
    ) -> None:
        return np_testing.assert_equal(array_a, array_b)

    def block(self, arrays: Sequence[np_array_t]) -> np_array_t:
        return np.block(arrays)

    def conjugate(self, array: np_array_t) -> np_array_t:
        return np.conjugate(array)

    def conjugate_transpose(self, array: np_array_t) -> np_array_t:
        return np.conjugate(self.transpose(array))

    def cos(self, array: np_array_t) -> np_array_t:
        return np.cos(array)

    def diag(self, array: np_array_t) -> np_array_t:
        return np.diag(array)

    def diagonal(
        self, array: np_array_t, axis1: int, axis2: int
    ) -> np_array_t:
        return np.diagonal(array, axis1, axis2)

    def eps(self) -> float:
        return np.finfo(self.dtype).eps

    def exp(self, array: np_array_t) -> np_array_t:
        return np.exp(array)

    def expand_dims(self, array: np_array_t, axis: int) -> np_array_t:
        return np.expand_dims(array, axis)

    def eye(self, size: int) -> np_array_t:
        return np.eye(size, dtype=self.dtype)

    def hstack(self, arrays: Sequence[np_array_t]) -> np_array_t:
        return np.hstack(arrays)

    def iscomplexobj(self, array: np_array_t) -> bool:
        return np.iscomplexobj(array)

    def isnan(self, array: np_array_t) -> np_array_t:
        return np.isnan(array)

    def isrealobj(self, array: np_array_t) -> bool:
        return np.isrealobj(array)

    def linalg_cholesky(self, array: np_array_t) -> np_array_t:
        return np.linalg.cholesky(array)

    def linalg_det(self, array: np_array_t) -> np_array_t:
        return np.linalg.det(array)

    def linalg_eigh(self, array: np_array_t) -> Tuple[np_array_t, np_array_t]:
        return np.linalg.eigh(array)

    def linalg_eigvalsh(
        self, array_x: np_array_t, array_y: Optional[array_t] = None
    ) -> np_array_t:
        if array_y is None:
            return np.linalg.eigvalsh(array_x)
        else:
            return np.vectorize(
                scipy.linalg.eigvalsh, signature="(m,m),(m,m)->(m)"
            )(array_x, array_y)

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

    def linalg_inv(self, array: np_array_t) -> np_array_t:
        return np.linalg.inv(array)

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

    def linalg_matrix_rank(self, array: np_array_t) -> int:
        return np.linalg.matrix_rank(array)

    def linalg_norm(
        self, array: np_array_t, *args: Any, **kwargs: Any
    ) -> np_array_t:
        return np.linalg.norm(array, *args, **kwargs)  # type: ignore

    def linalg_qr(self, array: np_array_t) -> Tuple[np_array_t, np_array_t]:
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
        self, array_a: np_array_t, array_b: np_array_t
    ) -> np_array_t:
        return np.linalg.solve(array_a, array_b)

    def linalg_solve_continuous_lyapunov(
        self, array_a: np_array_t, array_q: np_array_t
    ) -> np_array_t:
        return scipy.linalg.solve_continuous_lyapunov(array_a, array_q)

    def linalg_svd(
        self, array: np_array_t, *args: Any, **kwargs: Any
    ) -> Tuple[np_array_t, np_array_t, np_array_t]:
        return np.linalg.svd(array, *args, **kwargs)

    def log(self, array: np_array_t) -> np_array_t:
        return np.log(array)

    def logspace(self, *args: int) -> np_array_t:
        return np.logspace(*args, dtype=self.dtype)

    def ndim(self, array: np_array_t) -> int:
        return array.ndim

    def ones(self, shape: Sequence[int]) -> np_array_t:
        return np.ones(shape, self.dtype)

    def prod(self, array: np_array_t) -> float:
        return np.prod(array)  # type: ignore

    def random_normal(
        self, loc: float = 0.0, scale: float = 1.0, size: Sequence[int] = (1,)
    ) -> np_array_t:
        return self.array(np.random.normal(loc=loc, scale=scale, size=size))

    def random_randn(self, *dims: int) -> np_array_t:
        return self.array(np.random.randn(*dims))

    def random_uniform(self, size: Optional[int] = None) -> np_array_t:
        return self.array(np.random.uniform(size=size))

    def real(self, array: np_array_t) -> np_array_t:
        return np.real(array)

    def sin(self, array: np_array_t) -> np_array_t:
        return np.sin(array)

    def sinc(self, array: np_array_t) -> np_array_t:
        return np.sinc(array)

    def sort(self, array: np_array_t) -> np_array_t:
        return np.sort(array)

    def spacing(self, array: np_array_t) -> np_array_t:
        return np.spacing(array)  # type: ignore

    def sqrt(self, array: np_array_t) -> np_array_t:
        return np.sqrt(array)

    def squeeze(self, array: np_array_t) -> np_array_t:
        return np.squeeze(array)

    def sum(self, array: np_array_t, *args: Any, **kwargs: Any) -> np_array_t:
        return np.sum(array, *args, **kwargs)  # type: ignore

    def sym(self, array: np_array_t) -> np_array_t:
        return 0.5 * (array + self.transpose(array))

    def tan(self, array: np_array_t) -> np_array_t:
        return np.tan(array)

    def tanh(self, array: np_array_t) -> np_array_t:
        return np.tanh(array)

    def tensordot(
        self, a: np_array_t, b: np_array_t, axes: int = 2
    ) -> np_array_t:
        return np.tensordot(a, b, axes=axes)

    def tile(self, array: np_array_t, reps: int | Sequence[int]) -> np_array_t:
        return np.tile(array, reps)

    def trace(
        self, array: np_array_t, *args: tuple, **kwargs: dict
    ) -> np_array_t:
        return np.trace(array, *args, **kwargs)  # type: ignore

    def transpose(self, array: np_array_t) -> np_array_t:
        new_shape = list(range(self.ndim(array)))
        new_shape[-1], new_shape[-2] = new_shape[-2], new_shape[-1]
        return np.transpose(array, new_shape)

    def triu_indices(self, n: int, k: int = 0) -> np_array_t:
        return np.triu_indices(n, k)

    def vstack(self, arrays: Sequence[np_array_t]) -> np_array_t:
        return np.vstack(arrays)

    def where(self, condition: np_array_t) -> np_array_t:
        return np.where(condition)  # type: ignore

    def zeros(self, shape: Sequence[int]) -> np_array_t:
        return np.zeros(shape, dtype=self.dtype)

    def zeros_like(self, array: np_array_t) -> np_array_t:
        return np.zeros_like(array, dtype=self.dtype)
