from numbers import Number
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import scipy
import tensorflow as tf

from pymanopt.backends.backend import Backend, TupleOrList
from pymanopt.tools import (
    bisect_sequence,
    unpack_singleton_sequence_return_value,
)


# This allows to use multiple features present in numpy and other backends:
# - tranpose of matrices with x.T
# - type promotion between floats, ints and complex
# - correct broadcasting for tensors with different ndims (in particular for
#   matrix vector multiplication)
# for more details see documentation:
# https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/experimental_enable_numpy_behavior
tf.experimental.numpy.experimental_enable_numpy_behavior(prefer_float32=True)


def elementary_math_function(
    f: Callable[["TensorflowBackend", tf.Tensor], tf.Tensor],
) -> Callable[
    ["TensorflowBackend", Union[tf.Tensor, Number]], Union[tf.Tensor, Number]
]:
    def inner(
        self: "TensorflowBackend", x: Union[tf.Tensor, Number]
    ) -> Union[tf.Tensor, Number]:
        if isinstance(x, tf.Tensor):
            return f(self, x)
        else:
            return f(self, self.array(x)).numpy().item()

    inner.__doc__ = f.__doc__
    inner.__name__ = f.__name__
    return inner


class TensorflowBackend(Backend):
    ##########################################################################
    # Common attributes, properties and methods
    ##########################################################################
    array_t = tf.Tensor  # type: ignore
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
        return f"TensorflowBackend(dtype={self.dtype})"

    def to_real_backend(self) -> "TensorflowBackend":
        if self.is_dtype_real:
            return self
        if self.dtype == tf.complex64:
            return TensorflowBackend(dtype=tf.float32)
        elif self.dtype == tf.complex128:
            return TensorflowBackend(dtype=tf.float64)
        else:
            raise ValueError(f"dtype {self.dtype} is not supported")

    def to_complex_backend(self) -> "TensorflowBackend":
        if not self.is_dtype_real:
            return self
        if self.dtype == tf.float32:
            return TensorflowBackend(dtype=tf.complex64)
        elif self.dtype == tf.float64:
            return TensorflowBackend(dtype=tf.complex128)
        else:
            raise ValueError(f"dtype {self.dtype} is not supported")

    def _complex_to_real_dtype(self, complex_dtype: tf.DType) -> tf.DType:
        if complex_dtype == tf.complex64:
            return tf.float32
        elif complex_dtype == tf.complex128:
            return tf.float64
        else:
            raise ValueError(f"Provided dtype {complex_dtype} is not complex.")

    ##############################################################################
    # Autodiff methods
    ##############################################################################

    def _sanitize_gradient(self, tensor, grad):
        if grad is None:
            return tf.zeros_like(tensor)
        return grad

    def _sanitize_gradients(self, tensors, grads):
        return list(map(self._sanitize_gradient, tensors, grads))

    def prepare_function(self, function):
        return function

    def generate_gradient_operator(self, function, num_arguments):
        def gradient(*args):
            with tf.GradientTape() as tape:
                for arg in args:
                    tape.watch(arg)
                gradients = tape.gradient(function(*args), args)
                return self._sanitize_gradients(args, gradients)

        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(gradient)
        return gradient

    def generate_hessian_operator(self, function, num_arguments):
        def hessian_vector_product(*args):
            arguments, vectors = bisect_sequence(args)
            with (
                tf.GradientTape() as tape,
                tf.autodiff.ForwardAccumulator(
                    arguments, vectors
                ) as accumulator,
            ):
                for argument in arguments:
                    tape.watch(argument)
                gradients = tape.gradient(function(*arguments), arguments)
            return self._sanitize_gradients(
                arguments, accumulator.jvp(gradients)
            )

        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(
                hessian_vector_product
            )
        return hessian_vector_product

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

    def arange(
        self,
        start: int,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> tf.Tensor:
        if stop is None:
            return tf.range(start)
        if step is None:
            return tf.range(start, stop)
        return tf.range(start, stop, step)

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

    def array(self, array: Any) -> tf.Tensor:  # type: ignore
        if isinstance(array, tf.Tensor):
            if self.is_dtype_real and self.iscomplexobj(array):
                array = tf.math.real(array)
            array = tf.cast(array, dtype=self.dtype)
        return tf.convert_to_tensor(array, dtype=self.dtype)

    def assert_allclose(
        self,
        array_a: tf.Tensor,
        array_b: tf.Tensor,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        # if not isinstance(array_a, tf.Tensor):
        #     array_a = tf.constant(array_a, dtype=self.dtype)
        # if array_a.dtype != self.dtype:
        #     array_a = tf.cast(array_a, self.dtype)
        # if not isinstance(array_b, tf.Tensor):
        #     array_b = tf.constant(array_b, dtype=self.dtype)
        # if array_b.dtype != self.dtype:
        #     array_b = tf.cast(array_b, self.dtype)
        # tf.debugging.assert_near(array_a, array_b, rtol=rtol, atol=atol)
        def max_abs(x):
            return tf.math.reduce_max(tf.abs(x))

        assert self.allclose(array_a, array_b, rtol, atol), (
            "Arrays are not almost equal.\n"
            f"Max absolute difference: {max_abs(array_a - array_b)}"
            f" (atol={atol})\n"
            "Max relative difference: "
            f"{max_abs(array_a - array_b) / max_abs(array_b)}"
            f" (rtol={rtol})"
        )

    def assert_equal(
        self,
        array_a: tf.Tensor,
        array_b: tf.Tensor,
    ) -> None:
        tf.debugging.assert_equal(array_a, array_b)

    def concatenate(
        self, arrays: TupleOrList[tf.Tensor], axis: int = 0
    ) -> tf.Tensor:
        return tf.concat(arrays, axis)

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

    def imag(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.imag(array)

    def iscomplexobj(self, array: tf.Tensor) -> bool:
        return tf.experimental.numpy.iscomplexobj(array)

    @elementary_math_function
    def isnan(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.is_nan(array)

    def isrealobj(self, array: tf.Tensor) -> bool:
        return tf.experimental.numpy.isrealobj(array)

    def linalg_cholesky(self, array: tf.Tensor) -> tf.Tensor:
        return tf.linalg.cholesky(array)

    def linalg_det(self, array: tf.Tensor) -> tf.Tensor:
        return tf.linalg.det(array)

    def linalg_eigh(self, array: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        w, u = tf.linalg.eigh(array)
        return tf.math.real(w), u

    def linalg_eigvalsh(
        self, array_x: tf.Tensor, array_y: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        if array_y is None:
            return tf.math.real(tf.linalg.eigvalsh(array_x))
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
            return self.array(
                tf.linalg.logm(self.to_complex_backend().array(array))
            )

        w, v = tf.linalg.eigh(array)
        w = tf.expand_dims(tf.math.log(w), axis=-1)
        logmA = v @ (w * tf.linalg.adjoint(v))
        if array.dtype in {tf.float32, tf.float64}:
            return tf.math.real(logmA)
        return logmA

    def linalg_matrix_rank(self, array: tf.Tensor) -> int:
        return tf.linalg.matrix_rank(array).numpy().item()

    def linalg_norm(
        self,
        array: tf.Tensor,
        ord: Union[int, Literal["fro"], None] = None,
        axis: Union[int, TupleOrList[int], None] = None,
        keepdims: bool = False,
    ) -> tf.Tensor:
        if ord == "fro" or ord is None:
            ord = "euclidean"  # type: ignore
        return tf.math.real(
            tf.norm(array, ord=ord, axis=axis, keepdims=keepdims)
        )

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
        # if array_b.ndim < array_a.ndim:
        #     array_b = tf.expand_dims(array_b, -1)
        # sol = tf.linalg.solve(array_a, array_b)
        # return sol[..., 0] if array_b.ndim < array_a.ndim else sol

    def linalg_solve_continuous_lyapunov(
        self, array_a: tf.Tensor, array_q: tf.Tensor
    ) -> tf.Tensor:
        return self.array(
            scipy.linalg.solve_continuous_lyapunov(
                array_a.numpy(), array_q.numpy()
            )
        )

    def linalg_svd(
        self, array: tf.Tensor, full_matrices: bool = True
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        s, u, v = tf.linalg.svd(array, full_matrices=full_matrices)
        return u, s, self.conjugate_transpose(v)

    def linalg_svdvals(self, array: tf.Tensor) -> tf.Tensor:
        return tf.linalg.svd(array, compute_uv=False)

    @elementary_math_function
    def log(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.log(array)

    @elementary_math_function
    def log10(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.log(array) / tf.math.log(tf.constant(10.0))

    def logspace(self, start: float, stop: float, num: int) -> tf.Tensor:
        return tf.experimental.numpy.logspace(
            start, stop, num, dtype=self.dtype
        )

    def matvec(self, A: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        assert A.ndim >= 2
        if x.ndim == A.ndim - 1:
            return self.squeeze(A @ tf.expand_dims(x, -1))
        return tf.matmul(A, x)

    def matmul(self, A: tf.Tensor, B: tf.Tensor) -> tf.Tensor:
        return tf.matmul(A, B)

    def multieye(self, k: int, n: int) -> tf.Tensor:
        return tf.eye(n, batch_shape=[k], dtype=self.dtype)

    def ndim(self, array: tf.Tensor) -> int:
        return array.shape.rank

    def ones(self, shape: TupleOrList[int]) -> tf.Tensor:
        return tf.ones(shape, dtype=self.dtype)

    def ones_bool(self, shape: TupleOrList[int]) -> tf.Tensor:
        return tf.ones(shape, dtype=tf.bool)

    def polyfit(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        deg: int = 1,
        full: bool = False,
    ) -> Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
        assert x.ndim == y.ndim == 1
        x = tf.stack([x**i for i in range(deg + 1)], axis=-1)
        p = tf.squeeze(tf.linalg.lstsq(x, tf.reshape(y, (-1, 1))))
        if not full:
            return p
        res = tf.reduce_sum((y - x @ p) ** 2)
        return p, res

    def polyval(self, p: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        assert x.ndim == p.ndim == 1
        return tf.stack([x**i for i in range(p.shape[0])], axis=-1) @ p

    def prod(self, array: tf.Tensor) -> float:
        return tf.reduce_prod(array).numpy().item()

    def random_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Union[int, TupleOrList[int]] = 1,
    ) -> tf.Tensor:
        # pre-process the size
        if isinstance(size, int):
            new_size = (size,)
        elif size is None:
            new_size = (1,)
        else:
            new_size = size
        new_size = tf.constant(new_size)
        # sample
        if self.is_dtype_real:
            samples = tf.random.normal(
                shape=new_size, mean=loc, stddev=scale, dtype=self.dtype
            )
        else:
            real_dtype = self._complex_to_real_dtype(self.dtype)
            samples = tf.cast(
                tf.random.normal(shape=new_size, mean=loc, dtype=real_dtype),
                self.dtype,
            ) + 1j * tf.cast(
                tf.random.normal(shape=new_size, mean=loc, dtype=real_dtype),
                self.dtype,
            )
        # post-process
        return samples.numpy().item() if size is None else samples

    def random_uniform(self, size: Optional[int] = None) -> tf.Tensor:
        # pre-process the size
        if isinstance(size, int):
            new_size = (size,)
        elif size is None:
            new_size = (1,)
        else:
            new_size = size
        new_size = tf.constant(new_size)
        # sample
        if self.is_dtype_real:
            samples = tf.random.uniform(shape=new_size, dtype=self.dtype)
        else:
            real_dtype = self._complex_to_real_dtype(self.dtype)
            samples = tf.cast(
                tf.random.uniform(shape=new_size, dtype=real_dtype), self.dtype
            ) + 1j * tf.cast(
                tf.random.uniform(shape=new_size, dtype=real_dtype), self.dtype
            )
        # post-process
        return samples.numpy().item() if size is None else samples

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

    def sort(self, array: tf.Tensor, descending: bool = False) -> tf.Tensor:
        return tf.sort(
            array, direction="DESCENDING" if descending else "ASCENDING"
        )

    @elementary_math_function
    def sqrt(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.sqrt(array)

    def squeeze(self, array: tf.Tensor) -> tf.Tensor:
        return tf.squeeze(array)

    def stack(
        self, arrays: TupleOrList[tf.Tensor], axis: int = 0
    ) -> tf.Tensor:
        return tf.stack(arrays, axis=axis)

    def sum(
        self,
        array: tf.Tensor,
        axis: Union[int, TupleOrList[int], None] = None,
        keepdims: bool = False,
    ) -> tf.Tensor:
        # assert axis is None and not keepdims
        return tf.reduce_sum(array, axis=axis, keepdims=keepdims)

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
        self, array: tf.Tensor, reps: Union[int, TupleOrList[int]]
    ) -> tf.Tensor:
        return tf.tile(array, reps)

    def trace(self, array: tf.Tensor) -> tf.Tensor:
        return (
            tf.linalg.trace(array).numpy().item()
            if array.ndim == 2
            else tf.linalg.trace(array)
        )

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

    def zeros_bool(self, shape: TupleOrList[int]) -> tf.Tensor:
        return tf.zeros(shape, tf.bool)

    def zeros_like(self, array: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(array, dtype=self.dtype)
