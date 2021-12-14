import inspect

from pymanopt.manifolds.manifold import Manifold


class Function:
    def __init__(self, *, function, manifold, backend):
        if not callable(function):
            raise TypeError(f"Object {function} is not callable")
        if not backend.is_available():
            raise RuntimeError(f"Backend `{backend}' is not available")

        self._function = function
        self._backend = backend
        self._compiled_function = backend.compile_function(function)
        self._num_arguments = manifold.num_values

        self._egrad = None
        self._ehess = None

    def __str__(self):
        return "Function <{}>".format(self._backend)

    def compute_gradient(self):
        if self._egrad is None:
            self._egrad = self._backend.compute_gradient(
                self._function, self._num_arguments)
        return self._egrad

    def compute_hessian_vector_product(self):
        if self._ehess is None:
            self._ehess = self._backend.compute_hessian_vector_product(
                self._function, self._num_arguments)
        return self._ehess

    def __call__(self, *args, **kwargs):
        return self._compiled_function(*args, **kwargs)


def make_tracing_backend_decorator(Backend):
    """Create autodiff backend function decorator.

    A backend decorator factory to be used by autodiff backends to create a
    decorator for callables defined using the respective framework. The
    created decorator accepts a single argument ``manifold`` that specifies
    the domain (i.e., the manifold) of the decorated function:

      @decorator(manifold)
      def f(x):
          ...
    """
    def decorator(manifold):
        if not isinstance(manifold, Manifold):
            raise TypeError(
                "Backend decorator requires a manifold instance, got "
                f"{manifold}"
            )

        def inner(function):
            argspec = inspect.getfullargspec(function)
            if ((argspec.args and argspec.varargs) or
                    not (argspec.args or argspec.varargs) or
                    (argspec.varkw or argspec.kwonlyargs)):
                raise ValueError(
                    "Decorated function must only accept positional arguments "
                    "or a variable-length argument like *x"
                )
            return Function(
                function=function, manifold=manifold, backend=Backend()
            )
        return inner
    return decorator
