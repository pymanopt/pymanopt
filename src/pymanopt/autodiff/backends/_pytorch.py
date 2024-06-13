import functools
import warnings


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

    def _sanitize_gradient(self, tensor):
        if tensor.grad is None:
            return torch.zeros_like(tensor)
        return tensor.grad

    def _sanitize_gradients(self, tensors):
        return list(map(self._sanitize_gradient, tensors))

    @Backend._assert_backend_available
    def generate_gradient_operator(self, function, num_arguments):
        def gradient(*args):
            arguments = [arg.requires_grad_() for arg in args]
            function(*arguments).backward()
            return self._sanitize_gradients(arguments)

        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(gradient)
        return gradient

    @Backend._assert_backend_available
    def generate_hessian_operator(self, function, num_arguments):
        def hessian_vector_product(*args):
            arguments, vectors = bisect_sequence(args)
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
