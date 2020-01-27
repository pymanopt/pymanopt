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
from ...tools import flatten_arguments, group_return_values


class _PyTorchBackend(Backend):
    def __init__(self):
        super().__init__("PyTorch")

    @staticmethod
    def is_available():
        return torch is not None and torch.__version__ >= "0.4.1"

    @Backend._assert_backend_available
    def is_compatible(self, function, arguments):
        return callable(function)

    @Backend._assert_backend_available
    def compile_function(self, function, arguments):
        flattened_arguments = flatten_arguments(arguments)

        if len(flattened_arguments) == 1:
            @functools.wraps(function)
            def unary_function(argument):
                return function(torch.from_numpy(argument)).numpy()
            return unary_function

        @functools.wraps(function)
        def nary_function(arguments):
            return function(
                *map(torch.from_numpy, flatten_arguments(arguments))).numpy()
        return nary_function

    @Backend._assert_backend_available
    def compute_gradient(self, function, arguments):
        flattened_arguments = flatten_arguments(arguments)

        if len(flattened_arguments) == 1:
            @functools.wraps(function)
            def unary_gradient(argument):
                torch_argument = torch.from_numpy(argument)
                torch_argument.requires_grad_(True)
                function(torch_argument).backward()
                return torch_argument.grad.numpy()
            return unary_gradient

        def nary_gradient(arguments):
            torch_arguments = []
            for argument in flatten_arguments(arguments):
                torch_argument = torch.from_numpy(argument)
                torch_argument.requires_grad_()
                torch_arguments.append(torch_argument)
            function(*torch_arguments).backward()
            return [argument.grad.numpy() for argument in torch_arguments]
        return group_return_values(nary_gradient, arguments)

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
