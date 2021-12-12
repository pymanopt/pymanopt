import inspect

from pymanopt.manifolds.manifold import Manifold


class Function:
    def __init__(self, *, function, manifold, backend):
        self._validate_backend(backend)

        self._function = function
        self._manifold = manifold
        self._backend = backend

        self._num_arguments = self._get_number_of_arguments(manifold)
        self._compiled_function = None
        self._egrad = None
        self._ehess = None

        self._compile()

    def _get_number_of_arguments(self, manifold):
        point_layout = manifold.point_layout
        if hasattr(point_layout, "__iter__"):
            return sum(point_layout)
        return point_layout

    def __str__(self):
        return "Function <{}>".format(self._backend)

    def _validate_backend(self, backend):
        if not backend.is_available():
            raise ValueError("Backend `{}' is not available".format(
                backend)
            )

    def _compile(self):
        assert self._backend is not None
        if self._compiled_function is None:
            self._compiled_function = self._backend.compile_function(
                self._function
            )

    def compute_gradient(self):
        assert self._backend is not None
        if self._egrad is None:
            self._egrad = self._backend.compute_gradient(
                self._function, self._num_arguments)
        return self._egrad

    def compute_hessian_vector_product(self):
        assert self._backend is not None
        if self._ehess is None:
            self._ehess = self._backend.compute_hessian_vector_product(
                self._function, self._num_arguments)
        return self._ehess

    def __call__(self, *args, **kwargs):
        assert self._compiled_function is not None
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
