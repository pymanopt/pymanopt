import functools
import warnings

import numpy as np


try:
    import torch
except ImportError:
    torch = None
else:
    from torch import autograd

from ...tools import bisect_sequence, unpack_singleton_sequence_return_value
from ._backend import Backend


class PyTorchBackend(Backend):
    def __init__(self):
        super().__init__("PyTorch")

    @staticmethod
    def is_available():
        return torch is not None and torch.__version__ >= "0.4.1"

    @staticmethod
    def _from_numpy(array: np.ndarray):
        """Wrap numpy ndarray ``array`` in a torch tensor.

        Since torch does not support negative strides, we create a copy of the
        array to reset the strides in that case.
        """
        strides = np.array(array.strides)
        if np.any(strides < 0):
            warnings.warn(
                "PyTorch does not support numpy arrays with negative strides. "
                "Copying array to normalize strides."
            )
            array = array.copy()
        return torch.from_numpy(array)

    @Backend._assert_backend_available
    def prepare_function(self, function):
        @functools.wraps(function)
        def wrapper(*args):
            return function(*map(self._from_numpy, args)).numpy()

        return wrapper

    def _sanitize_gradient(self, tensor):
        if tensor.grad is None:
            return torch.zeros_like(tensor).numpy()
        return tensor.grad.numpy()

    def _sanitize_gradients(self, tensors):
        return list(map(self._sanitize_gradient, tensors))

    @Backend._assert_backend_available
    def generate_gradient_operator(self, function, num_arguments):
        def gradient(*args):
            arguments = [
                self._from_numpy(arg).requires_grad_() for arg in args
            ]
            function(*arguments).backward()
            return self._sanitize_gradients(arguments)

        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(gradient)
        return gradient

    @Backend._assert_backend_available
    def generate_hessian_operator(self, function, num_arguments):
        def hessian_vector_product(*args):
            arguments, vectors = bisect_sequence(
                list(map(self._from_numpy, args))
            )
            arguments = [argument.requires_grad_() for argument in arguments]
            gradients = autograd.grad(
                function(*arguments),
                arguments,
                create_graph=True,
                allow_unused=True,
            )
            dot_product = 0
            for gradient, vector in zip(gradients, vectors):
                dot_product += torch.tensordot(
                    gradient.conj(), vector, dims=gradient.ndim
                ).real
            dot_product.backward()
            return self._sanitize_gradients(arguments)

        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(
                hessian_vector_product
            )
        return hessian_vector_product
