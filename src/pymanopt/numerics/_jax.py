from typing import Any, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy as jscipy

from pymanopt.numerics.array_t import array_t
from pymanopt.numerics.core import NumericsBackend


class JaxNumericsBackend(NumericsBackend):
    _dtype: jnp.dtype

    def __init__(self, dtype=jnp.float64, random_seed: int = 42):
        self._dtype = dtype
        self._random_key = jax.random.key(random_seed)

    @property
    def dtype(self) -> jnp.dtype:
        return self._dtype

    def is_dtype_real(self):
        return jnp.issubdtype(self.dtype, jnp.floating)

    @property
    def DEFAULT_REAL_DTYPE(self):
        return jnp.array([1.0]).dtype

    @property
    def DEFAULT_COMPLEX_DTYPE(self):
        return jnp.array([1j]).dtype

    def __repr__(self):
        return f"JaxNumericsBackend(dtype={self.dtype})"

    ##############################################################################
    # Numerics functions
    ##############################################################################

    def abs(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.abs(array)

    def all(self, array: jnp.ndarray) -> bool:
        return jnp.all(array).item()

    def allclose(
        self,
        array_a: jnp.ndarray,
        array_b: jnp.ndarray,
        rtol: float = 1e-7,
        atol: float = 1e-10,
    ) -> bool:
        return jnp.allclose(array_a, array_b, rtol=rtol, atol=atol).item()

    def any(self, array: jnp.ndarray) -> bool:
        return jnp.any(array).item()

    def arange(self, *args: int) -> jnp.ndarray:
        return jnp.arange(*args)

    def arccos(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.arccos(array)

    def arccosh(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.arccosh(array)

    def arctan(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.arctan(array)

    def arctanh(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.arctanh(array)

    def argmin(self, array: jnp.ndarray):
        return jnp.argmin(array)

    def argsort(self, array: jnp.ndarray):
        return jnp.argsort(array)

    def array(self, array: array_t) -> jnp.ndarray:
        return jnp.asarray(array, dtype=self.dtype)

    def assert_allclose(
        self,
        array_a: jnp.ndarray,
        array_b: jnp.ndarray,
        rtol: float = 1e-7,
        atol: float = 1e-10,
    ) -> None:
        def max_abs(x):
            return jnp.max(jnp.abs(x))

        assert self.allclose(array_a, array_b, rtol, atol), (
            "Arrays are not almost equal.\n"
            f"Max absolute difference: {jnp.max(jnp.abs(array_a - array_b))}"
            f" (atol={atol})\n"
            "Max relative difference: "
            f"{max_abs(array_a - array_b) / max_abs(array_b)}"
            f" (rtol={rtol})"
        )

    def assert_almost_equal(
        self, array_a: jnp.ndarray, array_b: jnp.ndarray
    ) -> None:
        assert self.allclose(array_a, array_b)

    def assert_array_almost_equal(
        self, array_a: jnp.ndarray, array_b: jnp.ndarray
    ) -> None:
        assert self.allclose(array_a, array_b)

    def block(self, arrays: Sequence[jnp.ndarray]) -> jnp.ndarray:
        return jnp.block(arrays)

    def conjugate(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.conjugate(array)

    def conjugate_transpose(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.conjugate(self.transpose(array))

    def cos(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.cos(array)

    def diag(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.diag(array)

    def diagonal(
        self, array: jnp.ndarray, axis1: int, axis2: int
    ) -> jnp.ndarray:
        return jnp.diagonal(array, axis1, axis2)

    def eps(self) -> float:
        return jnp.finfo(self.dtype).eps

    def exp(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(array)

    def expand_dims(self, array: jnp.ndarray, axis: int) -> jnp.ndarray:
        return jnp.expand_dims(array, axis)

    def eye(self, size: int) -> jnp.ndarray:
        return jnp.eye(size, dtype=self.dtype)

    def hstack(self, arrays: Sequence[jnp.ndarray]) -> jnp.ndarray:
        return jnp.hstack(arrays)

    def iscomplexobj(self, array: jnp.ndarray) -> bool:
        return jnp.iscomplexobj(array)

    def isnan(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.isnan(array)

    def isrealobj(self, array: jnp.ndarray) -> bool:
        return jnp.isrealobj(array)

    def linalg_cholesky(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.cholesky(array)

    def linalg_det(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.det(array)

    def linalg_eigh(
        self, array: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.linalg.eigh(array)

    def linalg_eigvalsh(
        self, array_x: jnp.ndarray, array_y: Optional[array_t] = None
    ) -> jnp.ndarray:
        if array_y is None:
            return jnp.linalg.eigvalsh(array_x)
        else:
            return jnp.vectorize(
                jscipy.linalg.eigvalsh, signature="(m,m),(m,m)->(m)"
            )(array_x, array_y)

    def linalg_expm(
        self, array: jnp.ndarray, symmetric: bool = False
    ) -> jnp.ndarray:
        if not symmetric:
            # Scipy 1.9.0 added support for calling scipy.linalg.expm on stacked
            # matrices.
            # if pv.parse(scipy.version.version) >= pv.parse("1.9.0"):
            #     scipy_expm = scipy.linalg.expm
            # else:
            #     scipy_expm = jnp.vectorize(
            #         scipy.linalg.expm, signature="(m,m)->(m,m)"
            #     )
            return jscipy.linalg.expm(array)

        w, v = jnp.linalg.eigh(array)
        w = jnp.expand_dims(jnp.exp(w), axis=-1)
        expmA = v @ (w * self.conjugate_transpose(v))
        if jnp.isrealobj(array):
            return jnp.real(expmA)
        return expmA

    def linalg_inv(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.inv(array)

    def linalg_logm(
        self, array: jnp.ndarray, positive_definite: bool = False
    ) -> jnp.ndarray:
        if not positive_definite:
            return jnp.vectorize(jscipy.linalg.logm, signature="(m,m)->(m,m)")(
                array
            )

        w, v = jnp.linalg.eigh(array)
        w = jnp.expand_dims(jnp.log(w), axis=-1)
        logmA = v @ (w * self.conjugate_transpose(v))
        if jnp.isrealobj(array):
            return jnp.real(logmA)
        return logmA

    def linalg_matrix_rank(self, array: jnp.ndarray) -> int:
        return jnp.linalg.matrix_rank(array).item()

    def linalg_norm(
        self, array: jnp.ndarray, *args: Any, **kwargs: Any
    ) -> jnp.ndarray:
        return jnp.linalg.norm(array, *args, **kwargs)  # type: ignore

    def linalg_qr(self, array: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q, r = jnp.linalg.qr(array)

        # Compute signs or unit-modulus phase of entries of diagonal of r.
        s = jnp.diagonal(r, axis1=-2, axis2=-1).copy()
        # s[s == 0] = 1
        s = jnp.where(s == 0, 1, s)
        s = s / jnp.abs(s)
        s = jnp.expand_dims(s, axis=-1)
        # normalize q and r to have either 1 or unit-modulus on the diagonal of r
        q = q * self.transpose(s)
        r = r * jnp.conjugate(s)
        return q, r

    def linalg_solve(
        self, array_a: jnp.ndarray, array_b: jnp.ndarray
    ) -> jnp.ndarray:
        return jnp.linalg.solve(array_a, array_b)

    def linalg_solve_continuous_lyapunov(
        self, array_a: jnp.ndarray, array_q: jnp.ndarray
    ) -> jnp.ndarray:
        return jscipy.linalg.solve_continuous_lyapunov(array_a, array_q)

    def linalg_svd(
        self, array: jnp.ndarray, *args: Any, **kwargs: Any
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return jnp.linalg.svd(array, *args, **kwargs)

    def log(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(array)

    def logspace(self, *args: int) -> jnp.ndarray:
        return jnp.logspace(*args, dtype=self.dtype)

    def ndim(self, array: jnp.ndarray) -> int:
        return array.ndim

    def ones(self, shape: Sequence[int]) -> jnp.ndarray:
        return jnp.ones(shape, self.dtype)

    def prod(self, array: jnp.ndarray) -> float:
        return jnp.prod(array)  # type: ignore

    def random_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Union[int, Sequence[int]] = 1,
    ) -> jnp.ndarray:
        if isinstance(size, int):
            size = (size,)
        else:
            size = tuple(size)
        self._random_key, new_key = jax.random.split(self._random_key)
        return (
            scale
            * jax.random.normal(key=new_key, shape=size, dtype=self.dtype)
            + loc
        )

    def random_randn(self, *dims: int) -> jnp.ndarray:
        self._random_key, new_key = jax.random.split(self._random_key)
        return jax.random.normal(key=new_key, shape=dims, dtype=self.dtype)

    def random_uniform(self, size: Optional[int] = None) -> jnp.ndarray:
        self._random_key, new_key = jax.random.split(self._random_key)
        return jax.random.uniform(key=new_key, shape=size, dtype=self.dtype)

    def real(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.real(array)

    def sin(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.sin(array)

    def sinc(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.sinc(array)

    def sort(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.sort(array)

    def spacing(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.spacing(array)  # type: ignore

    def sqrt(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(array)

    def squeeze(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.squeeze(array)

    def sum(
        self, array: jnp.ndarray, *args: Any, **kwargs: Any
    ) -> jnp.ndarray:
        return jnp.sum(array, *args, **kwargs)  # type: ignore

    def sym(self, array: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * (array + self.transpose(array))

    def tan(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.tan(array)

    def tanh(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(array)

    def tensordot(
        self, a: jnp.ndarray, b: jnp.ndarray, axes: int = 2
    ) -> jnp.ndarray:
        return jnp.tensordot(a, b, axes=axes)

    def tile(
        self, array: jnp.ndarray, reps: int | Sequence[int]
    ) -> jnp.ndarray:
        return jnp.tile(array, reps)

    def trace(
        self, array: jnp.ndarray, *args: tuple, **kwargs: dict
    ) -> jnp.ndarray:
        return jnp.trace(array, *args, **kwargs)  # type: ignore

    def transpose(self, array: jnp.ndarray) -> jnp.ndarray:
        new_shape = list(range(self.ndim(array)))
        new_shape[-1], new_shape[-2] = new_shape[-2], new_shape[-1]
        return jnp.transpose(array, new_shape)

    def triu_indices(self, n: int, k: int = 0) -> jnp.ndarray:
        return jnp.triu_indices(n, k)

    def vstack(self, arrays: Sequence[jnp.ndarray]) -> jnp.ndarray:
        return jnp.vstack(arrays)

    def where(self, condition: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(condition)  # type: ignore

    def zeros(self, shape: Sequence[int]) -> jnp.ndarray:
        return jnp.zeros(shape, dtype=self.dtype)

    def zeros_like(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(array, dtype=self.dtype)
