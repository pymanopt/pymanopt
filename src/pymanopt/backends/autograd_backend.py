import functools

import autograd
import autograd.numpy as anp
from numpy import complex64, complex128, float32, float64

from pymanopt.backends.numpy_backend import NumpyBackend
from pymanopt.tools import (
    bisect_sequence,
    unpack_singleton_sequence_return_value,
)


def conjugate_result(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return list(map(anp.conj, function(*args, **kwargs)))  # type: ignore

    return wrapper


class AutogradBackend(NumpyBackend):
    ##########################################################################
    # Common attributes, properties and methods
    ##########################################################################

    def __init__(self, dtype: type = float64):
        super().__init__(dtype)

    def __repr__(self):
        return f"AutogradBackend(dtype={self.dtype})"

    def to_real_backend(self) -> "AutogradBackend":
        if self.is_dtype_real:
            return self
        if self.dtype == complex64:
            return AutogradBackend(dtype=float32)
        elif self.dtype == complex128:
            return AutogradBackend(dtype=float64)
        else:
            raise ValueError(f"dtype {self.dtype} is not supported")

    def to_complex_backend(self) -> "AutogradBackend":
        if not self.is_dtype_real:
            return self
        if self.dtype == float32:
            return AutogradBackend(dtype=complex64)
        elif self.dtype == float64:
            return AutogradBackend(dtype=complex128)
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
            return anp.sum(
                [
                    anp.real(anp.tensordot(gradient, vector, axes=vector.ndim))
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

    ##############################################################################
    # Numerics functions
    ##############################################################################

    # Since autograd thinly wraps numpy, all the classical numpy functions can
    # be used directly on arrays produced by autograd as long as the results are
    # not used in autodiff (which is the case for these numerics functions).
