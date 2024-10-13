import numpy.testing as np_testing
import tensorflow as tf

from pymanopt.numerics.array_t import array_t
from pymanopt.numerics.core import NumericsBackend


class TensorflowNumericsBackend(NumericsBackend):
    _dtype: tf.DType

    def __init__(self, dtype=tf.float64):
        self._dtype = dtype

    @property
    def dtype(self) -> tf.DType:
        return self._dtype

    def is_dtype_real(self):
        return tf.is_real(self.dtype)

    @property
    def DEFAULT_REAL_DTYPE(self):
        return tf.constant([1.0]).dtype

    @property
    def DEFAULT_COMPLEX_DTYPE(self):
        return tf.constant([1j]).dtype

    def __repr__(self):
        return f"TensorflowNumericsBackend(dtype={self.dtype})"

    ##############################################################################
    # Numerics functions
    ##############################################################################

    def abs(self, array: tf.Tensor) -> tf.Tensor:
        return tf.abs(array)

    def all(self, array: tf.Tensor) -> bool:
        return tf.experimental.numpy.all(array)

    def any(self, array: tf.Tensor) -> bool:
        return tf.experimental.numpy.any(array)

    def eps(self, dtype: tf.DType) -> float:
        return tf.experimental.numpy.finfo(dtype)

    def array(self, array: array_t) -> tf.Tensor:  # type: ignore
        return tf.convert_to_tensor(array)

    def assert_allclose(
        self, array_a: float | tf.Tensor, array_b: float | tf.Tensor
    ) -> None:
        np_testing.assert_allclose(
            array_a.numpy() if isinstance(array_a, tf.Tensor) else array_a,
            array_b.numpy() if isinstance(array_b, tf.Tensor) else array_b,
        )

    def assert_almost_equal(
        self, array_a: tf.Tensor, array_b: tf.Tensor
    ) -> None:
        np_testing.assert_almost_equal(array_a.numpy(), array_b.numpy())

    def assert_array_almost_equal(
        self, array_a: tf.Tensor, array_b: tf.Tensor
    ) -> None:
        np_testing.assert_array_almost_equal(array_a.numpy(), array_b.numpy())
