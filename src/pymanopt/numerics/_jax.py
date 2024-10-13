import jax.numpy as jnp
import numpy as np
import numpy.testing as np_testing

from pymanopt.numerics.array_t import array_t
from pymanopt.numerics.core import NumericsBackend


class JaxNumericsBackend(NumericsBackend):
    _dtype: jnp.dtype

    def __init__(self, dtype=jnp.float64):
        self._dtype = dtype

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

    def any(self, array: jnp.ndarray) -> bool:
        return jnp.any(array).item()

    def array(self, array: array_t) -> jnp.ndarray:  # type: ignore
        return jnp.array(array)

    def eps(self, dtype: jnp.dtype) -> float:
        return jnp.finfo(dtype)

    def assert_allclose(
        self, array_a: jnp.ndarray, array_b: jnp.ndarray
    ) -> None:
        np_testing.assert_allclose(np.array(array_a), np.array(array_b))

    def assert_almost_equal(
        self, array_a: jnp.ndarray, array_b: jnp.ndarray
    ) -> None:
        np_testing.assert_almost_equal(np.array(array_a), np.array(array_b))

    def assert_array_almost_equal(
        self, array_a: jnp.ndarray, array_b: jnp.ndarray
    ) -> None:
        np_testing.assert_array_almost_equal(
            np.array(array_a), np.array(array_b)
        )
