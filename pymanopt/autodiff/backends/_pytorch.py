"""
Module containing functions to differentiate functions using pytorch.
"""
import functools

try:
    import torch
except ImportError:
    torch = None
else:
    from torch import autograd

from ._backend import Backend
from .. import make_tracing_backend_decorator
from ...tools import flatten_arguments


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
    def _torch_tensor_to_numpy_ndarray(function):
        """Decorator which tries to transform the return value of a function
        from a torch tensor to a numpy array.
        """
        @functools.wraps(function)
        def wrapper(*args):
            value = function(*args)
            try:
                return value.numpy()
            except AttributeError:
                pass
            return value
        return wrapper

    @Backend._assert_backend_available
    def compile_function(self, function, arguments):
        flattened_arguments = flatten_arguments(arguments)
        if len(flattened_arguments) == 1:
            @self._torch_tensor_to_numpy_ndarray
            @functools.wraps(function)
            def unary_function(argument):
                return function(torch.from_numpy(argument))
            return unary_function

        @self._torch_tensor_to_numpy_ndarray
        @functools.wraps(function)
        def nary_function(arguments):
            return function(
                *map(torch.from_numpy, flatten_arguments(arguments)))
        return nary_function

    @Backend._assert_backend_available
    def compute_gradient(self, function, arguments):
        raise NotImplementedError

        def grad(x):
            x = torch.from_numpy(x)
            x.requires_grad_(True)
            function(x).backward()
            g = x.grad
            # See above.
            try:
                return g.numpy()
            except AttributeError:
                pass
            return g
        return grad

    @Backend._assert_backend_available
    def compute_hessian(self, objective, argument):
        raise NotImplementedError

        def hess(x, v):
            x = torch.from_numpy(x)
            v = torch.from_numpy(v)
            x.requires_grad_(True)
            fx = objective(x)
            grad_fx = autograd.grad(fx, x, create_graph=True)[0]
            grad_fx.matmul(v).backward()
            g = x.grad
            # See above.
            try:
                return g.numpy()
            except AttributeError:
                pass
            return g
        return hess


PyTorch = make_tracing_backend_decorator(_PyTorchBackend)
