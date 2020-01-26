"""
Module containing functions to differentiate functions using pytorch.
"""
try:
    import torch
except ImportError:
    torch = None
else:
    from torch import autograd

from ._backend import Backend
from .. import make_tracing_backend_decorator


class _PyTorchBackend(Backend):
    def __init__(self):
        super().__init__("PyTorch")

    @staticmethod
    def is_available():
        return torch is not None and torch.__version__ >= "0.4"

    @Backend._assert_backend_available
    def is_compatible(self, objective, argument):
        return callable(objective)

    # TODO: Add support for product manifolds.

    @Backend._assert_backend_available
    def compile_function(self, objective, argument):
        def func(x):
            # PyTorch unboxes scalars automatically, but we still need to get a
            # numpy view of the data when "compiling" gradients or Hessians.
            f = objective(torch.from_numpy(x))
            try:
                return f.numpy()
            except AttributeError:
                pass
            return f
        return func

    @Backend._assert_backend_available
    def compute_gradient(self, objective, argument):
        def grad(x):
            x = torch.from_numpy(x)
            x.requires_grad_(True)
            objective(x).backward()
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
