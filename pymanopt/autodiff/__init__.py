import inspect

from ..tools import flatten_args


class Function(object):
    def __str__(self):
        return "Function <{:s}>".format(str(self._backend))

    def __init__(self, function, arg, backend):
        self._function = function
        self._arg = arg
        self._backend = backend

        self._compiled_function = None
        self._egrad = None
        self._ehess = None

        self._verify_backend()
        self._compile()

    def _verify_backend(self):
        if not self._backend.is_available():
            raise ValueError("Backend `{:s}' is not available".format(
                str(self._backend)))
        if not self._backend.is_compatible(self._function, self._arg):
            raise ValueError("Backend `{:s}' is not compatible with cost "
                             "function of type `{:s}'".format(
                                 str(self._backend),
                                 self._function.__class__.__name__))

    def _compile(self):
        assert self._backend is not None
        if self._compiled_function is None:
            self._compiled_function = self._backend.compile_function(
                self._function, self._arg)

    def compute_gradient(self):
        assert self._backend is not None
        if self._egrad is None:
            self._egrad = self._backend.compute_gradient(self._function,
                                                         self._arg)
        return self._egrad

    def compute_hessian(self):
        assert self._backend is not None
        if self._ehess is None:
            self._ehess = self._backend.compute_hessian(self._function,
                                                        self._arg)
        return self._ehess

    def __call__(self, *args, **kwargs):
        assert self._compiled_function is not None
        return self._compiled_function(*args, **kwargs)


# TODO: Rename these two to `create_{tracing,graph}_backend_decorator'.

def make_function_decorator(Backend):
    """
    Creates a function decorator which can either by used as

      @decorator
      def f(x): pass

    or

      @decorator(3, 1)
      def f(x, y, z, w): pass

    to annotate a tracing-based autodiff function with how the arguments are
    conceptually grouped together.
    """
    def decorator(*args):
        if len(args) == 1 and callable(args[0]):
            (f,) = args
            num_args = len(inspect.getargspec(f).args)
            # We use a tuple of None to signal to the backend how many
            # arguments a function requires. We do this as early as possible so
            # as not to lose the information when wrapping `f' in one of our
            # various wrapper functions which often accept varargs and kwargs.
            return Function(f, arg=(None,) * num_args, backend=Backend())
        def inner(f):
            return Function(f, arg=args, backend=Backend())
        return inner
    return decorator


def make_function_decorator_with_argument(Backend):
    def decorator(*args):
        def inner(f):
            graph = f(*flatten_args(args))
            return Function(graph, arg=args, backend=Backend())
        return inner
    return decorator
