import inspect


class Function:
    def __str__(self):
        return "Function <{}>".format(self._backend)

    def __init__(self, function, args, backend):
        self._function = function
        self._args = args
        self._backend = backend

        self._compiled_function = None
        self._egrad = None
        self._ehess = None

        self._validate_backend()
        self._compile()

    def _validate_backend(self):
        if not self._backend.is_available():
            raise ValueError("Backend `{}' is not available".format(
                self._backend))
        if not self._backend.is_compatible(self._function, self._args):
            raise ValueError("Backend `{}' is not compatible with cost "
                             "function of type `{:s}'".format(
                                 self._backend,
                                 self._function.__class__.__name__))

    def _compile(self):
        assert self._backend is not None
        if self._compiled_function is None:
            self._compiled_function = self._backend.compile_function(
                self._function, self._args)

    def compute_gradient(self):
        assert self._backend is not None
        if self._egrad is None:
            self._egrad = self._backend.compute_gradient(
                self._function, self._args)
        return self._egrad

    def compute_hessian_vector_product(self):
        assert self._backend is not None
        if self._ehess is None:
            self._ehess = self._backend.compute_hessian_vector_product(
                self._function, self._args)
        return self._ehess

    def __call__(self, *args, **kwargs):
        assert self._compiled_function is not None
        return self._compiled_function(*args, **kwargs)


def make_tracing_backend_decorator(Backend):
    """Creates a function decorator which can either by used as

      @decorator
      def f(x):
          pass

    or

      @decorator(backend_specific_kwarg=...)
      def f(x):
          pass

    to annotate a tracing-based autodiff function with how the arguments are
    conceptually grouped together.
    """
    def decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            (function,) = args
            argspec = inspect.getfullargspec(function)
            if argspec.varargs or argspec.varkw or argspec.kwonlyargs:
                raise ValueError(
                    "Decorated function must only accept positional "
                    "arguments")
            return Function(function, args=tuple(argspec.args),
                            backend=Backend())

        if len(args) != 0:
            raise ValueError("Only keyword arguments allowed")

        def inner(function):
            return Function(function, args=args, backend=Backend(**kwargs))
        return inner
    return decorator


def make_graph_backend_decorator(Backend):
    def decorator(*args, **kwargs):
        def inner(function):
            graph = function(*args)
            return Function(graph, args=args, backend=Backend(**kwargs))
        return inner
    return decorator
