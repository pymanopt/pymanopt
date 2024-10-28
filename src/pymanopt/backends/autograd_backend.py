import functools

import autograd
import autograd.numpy as np

from pymanopt.backends.backend import Backend
from pymanopt.tools import (
    bisect_sequence,
    unpack_singleton_sequence_return_value,
)


def conjugate_result(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return list(map(np.conj, function(*args, **kwargs)))

    return wrapper


class AutogradBackend(Backend):
    ##########################################################################
    # Common attributes, properties and methods
    ##########################################################################
    array_t = np.ndarray

    def __init__(self, dtype=np.float64):
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
        return f"AutogradBackend(dtype={self.dtype})"

    def to_real_backend(self) -> "AutogradBackend":
        if self.is_dtype_real:
            return self
        if self.dtype == np.complex64:
            return AutogradBackend(dtype=np.float32)
        elif self.dtype == np.complex128:
            return AutogradBackend(dtype=np.float64)
        else:
            raise ValueError(f"dtype {self.dtype} is not supported")

    def to_complex_backend(self) -> "AutogradBackend":
        if not self.is_dtype_real:
            return self
        if self.dtype == np.float32:
            return AutogradBackend(dtype=np.complex64)
        elif self.dtype == np.float64:
            return AutogradBackend(dtype=np.complex128)
        else:
            raise ValueError(f"dtype {self.dtype} is not supported")

    ##############################################################################
    # Autodiff methods
    ##############################################################################

    def prepare_function(self, function):
        return function

    def generate_gradient_operator(self, function, num_arguments):
        gradient = conjugate_result(
            autograd.grad(function, argnum=list(range(num_arguments)))
        )
        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(gradient)
        return gradient

    @staticmethod
    def _hessian_vector_product(function, argnum):
        gradient = autograd.grad(function, argnum)

        def vector_dot_gradient(*args):
            *arguments, vectors = args
            gradients = gradient(*arguments)
            return np.sum(
                [
                    np.real(np.tensordot(gradient, vector, axes=vector.ndim))
                    for gradient, vector in zip(gradients, vectors)
                ]
            )

        return autograd.grad(vector_dot_gradient, argnum)

    def generate_hessian_operator(self, function, num_arguments):
        hessian_vector_product = conjugate_result(
            self._hessian_vector_product(
                function,
                argnum=list(range(num_arguments)),
            )
        )

        @functools.wraps(hessian_vector_product)
        def wrapper(*args):
            arguments, vectors = bisect_sequence(args)
            return hessian_vector_product(*arguments, vectors)

        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(wrapper)
        return wrapper