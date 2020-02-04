"""
Module containing functions to differentiate functions using pytorch.
"""
import functools
import warnings

import numpy as np
try:
    import torch
except ImportError:
    torch = None
else:
    from torch import autograd

from ._backend import Backend
from .. import make_tracing_backend_decorator
from ...tools import bisect_sequence, unpack_singleton_sequence_return_value


class _PyTorchBackend(Backend):
    def __init__(self):
        super().__init__("PyTorch")

    @staticmethod
    def is_available():
        return torch is not None and torch.__version__ >= "0.4.1"

    @Backend._assert_backend_available
    def is_compatible(self, function, arguments):
        return callable(function)

    @staticmethod
    def _from_numpy(array):
        """Wrap numpy ndarray ``array`` in a torch tensor. Since torch does not
        support negative strides, we create a copy of the array to reset the
        strides in that case.
        """
        strides = np.array(array.strides)
        if np.any(strides < 0):
            warnings.warn(
                "PyTorch does not support numpy arrays with negative strides. "
                "Copying array to normalize strides.")
            array = array.copy()
        return torch.from_numpy(array)

    @Backend._assert_backend_available
    def compile_function(self, function, arguments):
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
    def compute_gradient(self, function, arguments):
        def gradient(*args):
            torch_arguments = []
            for argument in args:
                torch_argument = self._from_numpy(argument)
                torch_argument.requires_grad_()
                torch_arguments.append(torch_argument)
            function(*torch_arguments).backward()
            return self._sanitize_gradients(torch_arguments)
        if len(arguments) == 1:
            return unpack_singleton_sequence_return_value(gradient)
        return gradient

    @Backend._assert_backend_available
    def compute_hessian_vector_product(self, function, arguments):
        def hessian_vector_product(*args):
            points, vectors = bisect_sequence(args)
            torch_arguments = []
            for point in points:
                torch_argument = self._from_numpy(point)
                torch_argument.requires_grad_()
                torch_arguments.append(torch_argument)
            torch_vectors = [self._from_numpy(vector) for vector in vectors]
            fx = function(*torch_arguments)
            fx.requires_grad_()
            gradients = autograd.grad(fx, torch_arguments, create_graph=True,
                                      allow_unused=True)
            dot_product = 0
            for gradient, vector in zip(gradients, torch_vectors):
                dot_product += torch.tensordot(
                    gradient, vector, dims=gradient.dim())
            dot_product.backward()
            return self._sanitize_gradients(torch_arguments)
        if len(arguments) == 1:
            return unpack_singleton_sequence_return_value(
                hessian_vector_product)
        return hessian_vector_product


PyTorch = make_tracing_backend_decorator(_PyTorchBackend)
