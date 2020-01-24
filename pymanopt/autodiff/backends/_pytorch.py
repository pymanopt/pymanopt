"""
Module containing functions to differentiate functions using pytorch.
"""
try:
    import torch
except ImportError:
    torch = None
else:
    from torch import autograd

from ._backend import Backend, assert_backend_available
from .. import make_tracing_backend_decorator


class PyTorchBackend(Backend):
    def __str__(self):
        return "pytorch"

    @staticmethod
    def is_available():
        # XXX: PyTorch 0.4 unified the Tensor and Variable API. Higher-order
        #      derivatives to compute Hessian-vector products were introduced
        #      in 0.2 so we should make that the first supported version.
        #      However, supporting both Tensor and Variable requires a bit more
        #      work that we'll skip for now.
        return torch is not None and torch.__version__ >= "0.4"

    @assert_backend_available
    def is_compatible(self, objective, argument):
        return callable(objective)

    # TODO: Add support for product manifolds.

    @assert_backend_available
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

    @assert_backend_available
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

    @assert_backend_available
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


PyTorch = make_tracing_backend_decorator(PyTorchBackend)
