import inspect
import typing

from pymanopt.manifolds.manifold import Manifold


class Function:
    def __init__(self, *, function, manifold, backend):
        if not callable(function):
            raise TypeError(f"Object {function} is not callable")
        if not backend.is_available():
            raise RuntimeError(f"Backend '{backend}' is not available")

        self._function = function
        self._backend = backend
        self._compiled_function = backend.compile_function(function)
        self._num_arguments = manifold.num_values

        self._egrad = None
        self._ehess = None

    def __str__(self):
        return f"Function <{self._backend}>"

    def compute_gradient(self):
        if self._egrad is None:
            self._egrad = self._backend.compute_gradient(
                self._function, self._num_arguments
            )
        return self._egrad

    def compute_hessian_vector_product(self):
        if self._ehess is None:
            self._ehess = self._backend.compute_hessian_vector_product(
                self._function, self._num_arguments
            )
        return self._ehess

    def __call__(self, *args, **kwargs):
        return self._compiled_function(*args, **kwargs)


def make_tracing_backend_decorator(Backend) -> typing.Callable:
    """Create function decorator for a backend.

    Function to create a backend decorator that is used to annotate a
    callable::

        decorator = make_tracing_backend_decorator(Backend)

        @decorator(manifold)
        def function(x):
            ...

    Args:
        Backend: a class implementing the backend interface defined by
            :class:`pymanopt.autodiff.backend._backend._Backend`.

    Returns:
        A new backend decorator.
    """

    def decorator(manifold):
        if not isinstance(manifold, Manifold):
            raise TypeError(
                "Backend decorator requires a manifold instance, got "
                f"{manifold}"
            )

        def inner(function):
            argspec = inspect.getfullargspec(function)
            if (
                (argspec.args and argspec.varargs)
                or not (argspec.args or argspec.varargs)
                or (argspec.varkw or argspec.kwonlyargs)
            ):
                raise ValueError(
                    "Decorated function must only accept positional arguments "
                    "or a variable-length argument like *x"
                )
            return Function(
                function=function, manifold=manifold, backend=Backend()
            )

        return inner

    return decorator
