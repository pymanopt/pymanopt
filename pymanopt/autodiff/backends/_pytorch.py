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
            def unary_gradient(argument):
                torch_argument = torch.from_numpy(argument)
                torch_argument.requires_grad_(True)
                function(torch_argument).backward()
                if torch_argument.grad is None:
                    return torch.zeros(torch_argument.shape).numpy()
                return torch_argument.grad.numpy()
            return unary_gradient

        def nary_gradient(arguments):
            torch_arguments = []
            for argument in flatten_arguments(arguments):
                torch_argument = torch.from_numpy(argument)
                torch_argument.requires_grad_()
                torch_arguments.append(torch_argument)
            function(*torch_arguments).backward()
            return_values = []
            for argument in torch_arguments:
                if argument.grad is None:
                    return_values.append(torch.zeros(argument.shape).numpy())
                else:
                    return_values.append(argument.grad.numpy())
            return return_values
        return group_return_values(nary_gradient, arguments)

    @Backend._assert_backend_available
    def compute_hessian(self, function, arguments):
        flattened_arguments = flatten_arguments(arguments)

        if len(flattened_arguments) == 1:
            def unary_hessian(point, vector):
                x = torch.from_numpy(point)
                v = torch.from_numpy(vector)
                x.requires_grad_(True)
                fx = function(x)
                (grad_fx,) = autograd.grad(fx, x, create_graph=True,
                                           allow_unused=True)
                (grad_fx * v).sum().backward()
                if x.grad is None:
                    return torch.zeros(x.shape).numpy()
                return x.grad.numpy()
            return unary_hessian

        def nary_hessian(points, vectors):
            xs = []
            for point in flatten_arguments(points):
                x = torch.from_numpy(point)
                x.requires_grad_(True)
                xs.append(x)
            vs = [torch.from_numpy(vector)
                  for vector in flatten_arguments(vectors)]
            fx = function(*xs)
            fx.requires_grad_(True)
            gradients = autograd.grad(fx, xs, create_graph=True,
                                      allow_unused=True)
            dot_product = 0
            for gradient, vector in zip(gradients, vs):
                dot_product += (gradient * vector).sum()
            dot_product.backward()
            return_values = []
            for x in xs:
                if x.grad is None:
                    return_values.append(torch.zeros(x.shape).numpy())
                else:
                    return_values.append(x.grad.numpy())
            return return_values
        return group_return_values(nary_hessian, arguments)


PyTorch = make_tracing_backend_decorator(_PyTorchBackend)
