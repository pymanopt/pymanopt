from numbers import Number
from typing import Any, Callable, Optional, Union

import numpy as np
import scipy
import tensorflow as tf

from pymanopt.numerics.array_t import array_t
from pymanopt.numerics.core import NumericsBackend, TupleOrList


def elementary_math_function(
    f: Callable[["TensorflowNumericsBackend", tf.Tensor], tf.Tensor],
) -> Callable[[Union[tf.Tensor, Number]], Union[tf.Tensor, Number]]:
    def inner(
        self: "TensorflowNumericsBackend", x: Union[tf.Tensor, Number]
    ) -> Union[tf.Tensor, Number]:
        if isinstance(x, tf.Tensor):
            return f(self, x)
        else:
            return f(self, self.array(x)).numpy().item()

    inner.__doc__ = f.__doc__
    inner.__name__ = f.__name__
    return inner


class TensorflowNumericsBackend(NumericsBackend):
    _dtype: tf.DType

    def __init__(self, dtype=tf.float64):
        self._dtype = dtype

    @property
    def dtype(self) -> tf.DType:
        return self._dtype

    @property
    def is_dtype_real(self):
        return self.dtype in {tf.float32, tf.float64}

    @staticmethod
    def DEFAULT_REAL_DTYPE():
        return tf.constant([1.0]).dtype

    @staticmethod
    def DEFAULT_COMPLEX_DTYPE():
        return tf.constant([1j]).dtype

    def __repr__(self):
        return f"TensorflowNumericsBackend(dtype={self.dtype})"

    def to_real_backend(self) -> "TensorflowNumericsBackend":
        if self.is_dtype_real:
            return self
        if self.dtype == tf.complex64:
            return TensorflowNumericsBackend(dtype=tf.float32)
        elif self.dtype == tf.complex128:
            return TensorflowNumericsBackend(dtype=tf.float64)
        else:
            raise ValueError(f"dtype {self.dtype} is not supported")

    def to_complex_backend(self) -> "TensorflowNumericsBackend":
        if not self.is_dtype_real:
            return self
        if self.dtype == tf.float32:
            return TensorflowNumericsBackend(dtype=tf.complex64)
        elif self.dtype == tf.float64:
            return TensorflowNumericsBackend(dtype=tf.complex128)
        else:
            raise ValueError(f"dtype {self.dtype} is not supported")

    ##############################################################################
    # Numerics functions
    ##############################################################################

    @elementary_math_function
    def abs(self, array: tf.Tensor) -> tf.Tensor:
        return tf.abs(array)

    def all(self, array: tf.Tensor) -> bool:
        return tf.reduce_all(tf.constant(array, dtype=tf.bool)).numpy().item()

    def allclose(
        self,
        array_a: tf.Tensor,
        array_b: tf.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        return tf.reduce_all(
            tf.abs(array_a - array_b) <= (atol + rtol * tf.abs(array_b))
        )

    def any(self, array: tf.Tensor) -> bool:
        return tf.reduce_any(tf.constant(array, dtype=tf.bool)).numpy().item()

    def arange(self, *args: int) -> tf.Tensor:
        return tf.range(*args, dtype=self.dtype)

    @elementary_math_function
    def arccos(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.acos(array)

    @elementary_math_function
    def arccosh(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.acosh(array)

    @elementary_math_function
    def arctan(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.atan(array)

    @elementary_math_function
    def arctanh(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.atanh(array)

    def argmin(self, array: tf.Tensor):
        return tf.argmin(array)

    def argsort(self, array: tf.Tensor):
        return tf.argsort(array)

    def array(self, array: array_t) -> tf.Tensor:  # type: ignore
        if isinstance(array, tf.Tensor):
            array = tf.cast(array, dtype=self.dtype)
        return tf.convert_to_tensor(array, dtype=self.dtype)

    def assert_allclose(
        self,
        array_a: tf.Tensor,
        array_b: tf.Tensor,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        if not isinstance(array_a, tf.Tensor):
            array_a = tf.constant(array_a, dtype=self.dtype)
        if array_a.dtype != self.dtype:
            array_a = tf.cast(array_a, self.dtype)
        if not isinstance(array_b, tf.Tensor):
            array_b = tf.constant(array_b, dtype=self.dtype)
        if array_b.dtype != self.dtype:
            array_b = tf.cast(array_b, self.dtype)
        tf.debugging.assert_near(array_a, array_b, rtol=rtol, atol=atol)

    def assert_equal(
        self,
        array_a: tf.Tensor,
        array_b: tf.Tensor,
    ) -> None:
        tf.debugging.assert_equal(array_a, array_b)

    def block(self, arrays: TupleOrList[tf.Tensor]) -> tf.Tensor:
        # TODO: implement actual block (where wr could give
        # arbitrarily nested lists of arrays)
        return tf.concat(arrays, axis=0)

    @elementary_math_function
    def conjugate(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.conj(array)

    @elementary_math_function
    def cos(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.cos(array)

    def diag(self, array: tf.Tensor) -> tf.Tensor:
        return tf.linalg.diag(array)

    def diagonal(self, array: tf.Tensor, axis1: int, axis2: int) -> tf.Tensor:
        # TODO: check correctness
        return tf.linalg.diag_part(array)

    def eps(self) -> float:
        return tf.experimental.numpy.finfo(self.dtype).eps

    @elementary_math_function
    def exp(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.exp(array)

    def expand_dims(self, array: tf.Tensor, axis: int) -> tf.Tensor:
        return tf.expand_dims(array, axis)

    def eye(self, size: int) -> tf.Tensor:
        return tf.eye(size, dtype=self.dtype)

    def hstack(self, arrays: TupleOrList[tf.Tensor]) -> tf.Tensor:
        return tf.concat(arrays, axis=1)

    def iscomplexobj(self, array: tf.Tensor) -> bool:
        return tf.is_complex(array)

    @elementary_math_function
    def isnan(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.is_nan(array)

    def isrealobj(self, array: tf.Tensor) -> bool:
        return not tf.is_complex(array)

    def linalg_cholesky(self, array: tf.Tensor) -> tf.Tensor:
        return tf.linalg.cholesky(array)

    def linalg_det(self, array: tf.Tensor) -> tf.Tensor:
        return tf.linalg.det(array)

    def linalg_eigh(self, array: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return tf.linalg.eigh(array)

    def linalg_eigvalsh(
        self, array_x: tf.Tensor, array_y: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        if array_y is None:
            return tf.linalg.eigvalsh(array_x)
        else:
            return self.array(
                np.vectorize(
                    scipy.linalg.eigvalsh, signature="(m,m),(m,m)->(m)"
                )(array_x.numpy(), array_y.numpy())
            )

    def linalg_expm(
        self, array: tf.Tensor, symmetric: bool = False
    ) -> tf.Tensor:
        if not symmetric:
            return tf.linalg.expm(array)

        w, v = tf.linalg.eigh(array)
        w = tf.expand_dims(tf.exp(w), axis=-1)
        expmA = v @ (w * tf.linalg.adjoint(v))
        if array.dtype in {tf.float32, tf.float64}:
            return tf.math.real(expmA)
        return expmA

    def linalg_inv(self, array: tf.Tensor) -> tf.Tensor:
        return tf.linalg.inv(array)

    def linalg_logm(
        self, array: tf.Tensor, positive_definite: bool = False
    ) -> tf.Tensor:
        if not positive_definite:
            return tf.linalg.logm(array)

        w, v = tf.linalg.eigh(array)
        w = tf.expand_dims(tf.math.log(w), axis=-1)
        logmA = v @ (w * tf.linalg.adjoint(v))
        if array.dtype in {tf.float32, tf.float64}:
            return tf.math.real(logmA)
        return logmA

    def linalg_matrix_rank(self, array: tf.Tensor) -> int:
        return tf.linalg.matrix_rank(array).numpy().item()

    def linalg_norm(
        self, array: tf.Tensor, *args: Any, **kwargs: Any
    ) -> tf.Tensor:
        if len(args) > 0 and args[0] == "fro":
            args = ("euclidean",) + args[1:]
        elif kwargs.get("ord", None) == "fro":
            kwargs["ord"] = "euclidean"
        return tf.norm(array, *args, **kwargs)

    def linalg_qr(self, array: tf.Tensor) -> tf.Tensor:
        q, r = tf.linalg.qr(array)
        # Compute signs or unit-modulus phase of entries of diagonal of r.
        s = tf.identity(tf.linalg.diag_part(r))
        s = tf.where(tf.equal(s, 0.0), tf.ones_like(s), s)
        s = s / tf.cast(tf.abs(s), dtype=self.dtype)
        s = tf.expand_dims(s, axis=-1)
        # normalize q and r to have either 1 or unit-modulus on the diagonal of r
        q = q * self.transpose(s)
        r = r * self.conjugate(s)
        return q, r

    def linalg_solve(
        self, array_a: tf.Tensor, array_b: tf.Tensor
    ) -> tf.Tensor:
        return tf.linalg.solve(array_a, array_b)

    def linalg_solve_continuous_lyapunov(
        self, array_a: tf.Tensor, array_q: tf.Tensor
    ) -> tf.Tensor:
        return self.array(
            scipy.linalg.solve_continuous_lyapunov(
                array_a.numpy(), array_q.numpy()
            )
        )

    def linalg_svd(
        self, array: tf.Tensor, *args, **kwargs
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return tf.linalg.svd(array, *args, **kwargs)

    @elementary_math_function
    def log(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.log(array)

    def logspace(self, *args: int) -> tf.Tensor:
        return tf.experimental.numpy.logspace(*args, dtype=self.dtype)

    def ndim(self, array: tf.Tensor) -> int:
        return array.shape.rank

    def ones(self, shape: TupleOrList[int]) -> tf.Tensor:
        return tf.ones(shape, dtype=self.dtype)

    def prod(self, array: tf.Tensor) -> float:
        return tf.reduce_prod(array).numpy().item()

    def random_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Union[int, TupleOrList[int]] = 1,
    ) -> tf.Tensor:
        if isinstance(size, int):
            size = (size,)
        size = tf.constant(size)
        if self.is_dtype_real:
            return tf.random.normal(
                shape=size, mean=loc, stddev=scale, dtype=self.dtype
            )
        else:
            real_dtype = tf.experimental.numpy.finfo(self.dtype).dtype
            return tf.cast(
                tf.random.normal(shape=size, mean=loc, dtype=real_dtype),
                self.dtype,
            ) + 1j * tf.cast(
                tf.random.normal(shape=size, mean=loc, dtype=real_dtype),
                self.dtype,
            )

    def random_uniform(self, size: Optional[int] = None) -> tf.Tensor:
        if self.is_dtype_real:
            return tf.random.uniform(shape=size, dtype=self.dtype)
        else:
            return tf.cast(
                tf.random.uniform(shape=size), self.dtype
            ) + 1j * tf.cast(tf.random.uniform(shape=size), self.dtype)

    @elementary_math_function
    def real(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.real(array)

    def reshape(
        self, array: tf.Tensor, newshape: TupleOrList[int]
    ) -> tf.Tensor:
        return tf.reshape(array, newshape)

    @elementary_math_function
    def sin(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.sin(array)

    @elementary_math_function
    def sinc(self, array: tf.Tensor) -> tf.Tensor:
        return tf.experimental.numpy.sinc(array)

    def sort(self, array: tf.Tensor) -> tf.Tensor:
        return tf.sort(array)

    @elementary_math_function
    def sqrt(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.sqrt(array)

    def squeeze(self, array: tf.Tensor) -> tf.Tensor:
        return tf.squeeze(array)

    def sum(self, array: tf.Tensor, *args: Any, **kwargs: Any) -> tf.Tensor:
        return tf.reduce_sum(array, *args, **kwargs)

    @elementary_math_function
    def tan(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.tan(array)

    @elementary_math_function
    def tanh(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.tanh(array)

    def tensordot(
        self, a: tf.Tensor, b: tf.Tensor, axes: int = 2
    ) -> tf.Tensor:
        return tf.tensordot(a, b, axes=axes)

    def tile(
        self, array: tf.Tensor, reps: int | TupleOrList[int]
    ) -> tf.Tensor:
        return tf.tile(array, reps)

    def trace(self, array: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return tf.linalg.trace(array, *args, **kwargs)

    def transpose(self, array: tf.Tensor) -> tf.Tensor:
        perm = list(range(self.ndim(array)))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        return tf.transpose(array, perm)

    def triu(self, array: tf.Tensor, k: int = 0) -> tf.Tensor:
        return tf.experimental.numpy.triu(array, k)

    def vstack(self, arrays: TupleOrList[tf.Tensor]) -> tf.Tensor:
        return tf.concat(arrays, axis=0)

    def where(
        self,
        condition: tf.Tensor,
        x: Optional[tf.Tensor] = None,
        y: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        if x is None and y is None:
            return tf.where(condition)
        elif x is not None and y is not None:
            return tf.where(condition, x, y)
        else:
            raise ValueError(
                f"Both x and y have to be specified but are respectively {x} and {y}"
            )

    def zeros(self, shape: TupleOrList[int]) -> tf.Tensor:
        return tf.zeros(shape, dtype=self.dtype)

    def zeros_like(self, array: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(array, dtype=self.dtype)
