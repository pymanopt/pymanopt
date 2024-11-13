__all__ = ["Function", "autograd", "jax", "numpy", "pytorch", "tensorflow"]

import inspect
from importlib import import_module
from typing import Any, Callable

from pymanopt.backends import Backend
from pymanopt.manifolds.manifold import Manifold


class Function:
    def __init__(
        self, *, function: Callable, manifold: Manifold, backend: Backend
    ):
        if not callable(function):
            raise TypeError(f"Object {function} is not callable")

        self._original_function = function
        self._backend = backend
        self._function = backend.prepare_function(function)
        self._num_arguments = manifold.num_values

        self._gradient = None
        self._hessian = None

    def __str__(self):
        return f"Function <{self._backend}>"

    @property
    def backend(self):
        return self._backend

    def get_gradient_operator(self):
        if self._gradient is None:
            self._gradient = self._backend.generate_gradient_operator(
                self._original_function, self._num_arguments
            )
        return self._gradient

    def get_hessian_operator(self):
        if self._hessian is None:
            self._hessian = self._backend.generate_hessian_operator(
                self._original_function, self._num_arguments
            )
        return self._hessian

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)


def _only_one_true(*args):
    return sum(args) == 1


def decorator_factory(
    module: str, backend_class: str
) -> Callable[[Manifold], Callable[[Callable[..., Any]], Function]]:
    def decorator(
        manifold: Manifold,
    ) -> Callable[[Callable[..., Any]], Function]:
        assert isinstance(manifold, Manifold)

        def inner(cost: Callable[..., Any]) -> Function:
            argspec = inspect.getfullargspec(cost)
            assert (
                _only_one_true(bool(argspec.args), bool(argspec.varargs))
                and not argspec.varkw
                and not argspec.kwonlyargs
            ), (
                "Decorated function must only accept positional arguments "
                "or a variable-length argument like *x"
            )
            backend = getattr(
                import_module(
                    f"pymanopt.backends.{module}",
                ),
                backend_class,
            )()
            return Function(function=cost, manifold=manifold, backend=backend)

        return inner

    return decorator


numpy = decorator_factory("numpy_backend", "NumpyBackend")
jax = decorator_factory("jax_backend", "JaxBackend")
pytorch = decorator_factory("pytorch_backend", "PytorchBackend")
autograd = decorator_factory("autograd_backend", "AutogradBackend")
tensorflow = decorator_factory("tensorflow_backend", "TensorflowBackend")
