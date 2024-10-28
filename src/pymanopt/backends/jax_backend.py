import functools
from numbers import Number
from typing import Literal, Optional, Union

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
import scipy.linalg

from pymanopt.backends.backend import Backend, TupleOrList
from pymanopt.tools import (
    bisect_sequence,
    unpack_singleton_sequence_return_value,
)


# for backward compatibility with older versions of jax
try:
    from jax import config
except ImportError:
    from jax.config import config  # type: ignore

config.update("jax_enable_x64", True)


def conjugate_result(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return list(map(jnp.conj, function(*args, **kwargs)))

    return wrapper


def to_ndarray(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return list(map(np.asarray, function(*args, **kwargs)))

    return wrapper


class JaxBackend(Backend):
    array_t = jnp.ndarray
    _dtype: jnp.dtype

    ##########################################################################
    # Common attributes, properties and methods
    ##########################################################################
    def __init__(self, dtype=jnp.float64, random_seed: int = 42):
        self._dtype = dtype
        self._random_key = jax.random.key(random_seed)

    def _gen_1_random_key(self):
        self._random_key, new_key = jax.random.split(self._random_key)
        return new_key

    def _gen_2_random_keys(
        self,
    ):
        self._random_key, *new_keys = jax.random.split(self._random_key, 3)
        return new_keys

    @property
    def dtype(self) -> jnp.dtype:
        return self._dtype

    @property
    def is_dtype_real(self):
        return jnp.issubdtype(self.dtype, jnp.floating)

    @staticmethod
    def DEFAULT_REAL_DTYPE():
        return jnp.array([1.0]).dtype

    @staticmethod
    def DEFAULT_COMPLEX_DTYPE():
        return jnp.array([1j]).dtype

    def __repr__(self):
        return f"JaxBackend(dtype={self.dtype})"

    def to_real_backend(self) -> "JaxBackend":
        if self.is_dtype_real:
            return self
        if self.dtype == jnp.complex64:
            return JaxBackend(dtype=jnp.float32)
        elif self.dtype == jnp.complex128:
            return JaxBackend(dtype=jnp.float64)
        else:
            raise ValueError(f"dtype {self.dtype} is not supported")

    def to_complex_backend(self) -> "JaxBackend":
        if not self.is_dtype_real:
            return self
        if self.dtype == jnp.float32:
            return JaxBackend(dtype=jnp.complex64)
        elif self.dtype == jnp.float64:
            return JaxBackend(dtype=jnp.complex128)
        else:
            raise ValueError(f"dtype {self.dtype} is not supported")

    ##############################################################################
    # Autodiff methods
    ##############################################################################

    def prepare_function(self, function):
        return function

    def generate_gradient_operator(self, function, num_arguments):
        gradient = to_ndarray(
            conjugate_result(jax.grad(function, argnums=range(num_arguments)))
        )
        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(gradient)
        return gradient

    def generate_hessian_operator(self, function, num_arguments):
        @to_ndarray
        @conjugate_result
        def hessian_vector_product(arguments, vectors):
            return jax.jvp(
                jax.grad(function, argnums=range(num_arguments)),
                arguments,
                vectors,
            )[1]

        @functools.wraps(hessian_vector_product)
        def wrapper(*args):
            arguments, vectors = bisect_sequence(args)
            return hessian_vector_product(arguments, vectors)

        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(wrapper)
        return wrapper

    ##############################################################################
    # Numerics functions
    ##############################################################################

    def abs(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.abs(array)

    def all(self, array: jnp.ndarray) -> bool:
        return jnp.all(jnp.array(array, dtype=bool)).item()

    def allclose(
        self,
        array_a: jnp.ndarray,
        array_b: jnp.ndarray,
        rtol: float = 1e-7,
        atol: float = 1e-10,
    ) -> bool:
        return jnp.allclose(array_a, array_b, rtol=rtol, atol=atol).item()

    def any(self, array: jnp.ndarray) -> bool:
        return jnp.any(jnp.array(array, dtype=bool)).item()

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

    def array(self, array: array_t) -> jnp.ndarray:  # type: ignore
        return jnp.asarray(array, dtype=self.dtype)

    def assert_allclose(
        self,
        array_a: jnp.ndarray,
        array_b: jnp.ndarray,
        rtol: float = 1e-6,
        atol: float = 1e-6,
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

    def concatenate(
        self, arrays: TupleOrList[jnp.ndarray], axis: int = 0
    ) -> jnp.ndarray:
        return jnp.concatenate(arrays, axis)

    def conjugate(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.conjugate(array)

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

    def hstack(self, arrays: TupleOrList[jnp.ndarray]) -> jnp.ndarray:
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
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.linalg.eigh(array)

    def linalg_eigvalsh(
        self, array_x: jnp.ndarray, array_y: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        if array_y is None:
            return jscipy.linalg.eigh(array_x, array_y, eigvals_only=True)
        else:
            # the generalized eigen value problem is only supported in scipy
            # for the moment.
            return jnp.asarray(
                np.vectorize(
                    pyfunc=scipy.linalg.eigvalsh, signature="(m,m),(m,m)->(m)"
                )(np.asarray(array_x), np.asarray(array_y)),
                dtype=self.dtype,
            )

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
            return jnp.asarray(
                np.vectorize(scipy.linalg.logm, signature="(m,m)->(m,m)")(
                    np.asarray(array)
                ),
                dtype=self.dtype,
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
        self,
        array: jnp.ndarray,
        ord: Union[int, Literal["fro"], None] = None,
        axis: Union[int, TupleOrList[int], None] = None,
        keepdims: bool = False,
    ) -> jnp.ndarray:
        return jnp.linalg.norm(array, ord=ord, axis=axis, keepdims=keepdims)

    def linalg_qr(self, array: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        q, r = jnp.linalg.qr(array)

        # Compute signs or unit-modulus phase of entries of diagonal of r.
        s = jnp.diagonal(r, axis1=-2, axis2=-1).copy()
        s = jnp.where(s == 0.0, 1.0, s)
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
        return jnp.asarray(
            scipy.linalg.solve_continuous_lyapunov(
                np.asarray(array_a), np.asarray(array_q)
            )
        )

    def linalg_svd(
        self,
        array: jnp.ndarray,
        full_matrices: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return jnp.linalg.svd(array, full_matrices=full_matrices)  # type: ignore

    def linalg_svdvals(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.svdvals(array)

    def log(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(array)

    def log10(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.log10(array)

    def logspace(self, *args: int) -> jnp.ndarray:
        return jnp.logspace(*args, dtype=self.dtype)

    def ndim(self, array: jnp.ndarray) -> int:
        return array.ndim

    def ones(self, shape: TupleOrList[int]) -> jnp.ndarray:
        return jnp.ones(shape, self.dtype)

    def ones_bool(self, shape: TupleOrList[int]) -> jnp.ndarray:
        return jnp.ones(shape, bool)

    def polyfit(
        self, x: jnp.ndarray, y: jnp.ndarray, deg: int = 1, full: bool = False
    ) -> Union[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        return jnp.polyfit(x, y, deg, full=full)  # type: ignore

    def polyval(self, p: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.polyval(p, x)

    def prod(self, array: jnp.ndarray) -> float:
        return jnp.prod(array)  # type: ignore

    def random_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Union[int, TupleOrList[int], None] = None,
    ) -> jnp.ndarray:
        if isinstance(size, int):
            size = [size]
        elif size is None:
            size = ()
        if self.is_dtype_real:
            new_key = self._gen_1_random_key()
            return (
                scale
                * jax.random.normal(key=new_key, shape=size, dtype=self.dtype)
                + loc
            )
        else:
            real_dtype = jnp.finfo(self.dtype).dtype
            new_key_1, new_key_2 = self._gen_2_random_keys()
            return (
                scale
                * jax.random.normal(
                    key=new_key_1, shape=size, dtype=real_dtype
                )
                + loc
            ) + 1j * (
                scale
                * jax.random.normal(
                    key=new_key_2, shape=size, dtype=real_dtype
                )
                + loc
            )

    def random_uniform(
        self, size: Union[int, TupleOrList[int], None] = None
    ) -> jnp.ndarray:
        if isinstance(size, int):
            size = (size,)
        elif size is None:
            size = ()

        if self.is_dtype_real:
            new_key = self._gen_1_random_key()
            return jax.random.uniform(
                key=new_key, shape=size, dtype=self.dtype
            )
        else:
            real_dtype = jnp.finfo(self.dtype).dtype
            new_key_1, new_key_2 = self._gen_2_random_keys()
            return jax.random.uniform(
                key=new_key_1, shape=size, dtype=real_dtype
            ) + 1j * jax.random.uniform(
                key=new_key_2, shape=size, dtype=real_dtype
            )

    def real(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.real(array)

    def reshape(
        self, array: jnp.ndarray, newshape: TupleOrList[int]
    ) -> jnp.ndarray:
        return jnp.reshape(array, newshape)

    def sin(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.sin(array)

    def sinc(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.sinc(array)

    def sort(
        self, array: jnp.ndarray, descending: bool = False
    ) -> jnp.ndarray:
        return jnp.sort(array, descending=descending)

    def spacing(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(np.spacing(np.asarray(array)))

    def sqrt(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(array)

    def squeeze(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.squeeze(array)

    def stack(
        self, arrays: TupleOrList[jnp.ndarray], axis: int = 0
    ) -> jnp.ndarray:
        return jnp.stack(arrays, axis)

    def sum(
        self,
        array: jnp.ndarray,
        axis: Union[int, TupleOrList[int], None] = None,
        keepdims: bool = False,
    ) -> jnp.ndarray:
        return jnp.sum(array, axis=axis, keepdims=keepdims)

    def tan(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.tan(array)

    def tanh(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(array)

    def tensordot(
        self, a: jnp.ndarray, b: jnp.ndarray, axes: int = 2
    ) -> jnp.ndarray:
        return jnp.tensordot(a, b, axes=axes)

    def tile(
        self, array: jnp.ndarray, reps: Union[int, TupleOrList[int]]
    ) -> jnp.ndarray:
        return jnp.tile(array, reps)

    def trace(self, array: jnp.ndarray) -> Union[Number, jnp.ndarray]:
        return (
            jnp.trace(array).item()
            if array.ndim == 2
            else jnp.trace(array, axis1=-2, axis2=-1)
        )

    def transpose(self, array: jnp.ndarray) -> jnp.ndarray:
        new_shape = list(range(self.ndim(array)))
        new_shape[-1], new_shape[-2] = new_shape[-2], new_shape[-1]
        return jnp.transpose(array, new_shape)

    def triu(self, array: jnp.ndarray, k: int = 0) -> jnp.ndarray:
        return jnp.triu(array, k)

    def vstack(self, arrays: TupleOrList[jnp.ndarray]) -> jnp.ndarray:
        return jnp.vstack(arrays)

    def where(
        self,
        condition: jnp.ndarray,
        x: Optional[jnp.ndarray] = None,
        y: Optional[jnp.ndarray] = None,
    ) -> Union[jnp.ndarray, tuple[jnp.ndarray, ...]]:
        if x is None and y is None:
            return jnp.where(condition)
        elif x is not None and y is not None:
            return jnp.where(condition, x, y)
        else:
            raise ValueError(
                f"Both x and y have to be specified but are respectively {x} and {y}"
            )

    def zeros(self, shape: TupleOrList[int]) -> jnp.ndarray:
        return jnp.zeros(shape, dtype=self.dtype)

    def zeros_bool(self, shape: TupleOrList[int]) -> jnp.ndarray:
        return jnp.zeros(shape, bool)

    def zeros_like(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(array, dtype=self.dtype)
