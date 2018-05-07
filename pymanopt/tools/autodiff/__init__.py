from ._callable import CallableBackend
from ._autograd import AutogradBackend
from ._pytorch import PyTorchBackend
from ._theano import TheanoBackend
from ._tensorflow import TensorflowBackend

_BACKENDS = (CallableBackend, AutogradBackend, PyTorchBackend, TheanoBackend,
             TensorflowBackend)
__all__ = [Backend.__name__ for Backend in _BACKENDS]


class Function(object):
    def __init__(self, function, argument=None):
        if isinstance(function, Function):
            raise ValueError("Cannot wrap Function instance in Function")
        self._function = function
        self._arg = argument

        self._backend = self._determine_backend(function, argument)
        self._compile()

    @staticmethod
    def _determine_backend(function, argument):
        backend = None

        if hasattr(function, "backend"):
            backend = function.backend
            if not backend.is_compatible(function, argument):
                raise ValueError("Backend `{:s}' not compatible with cost "
                                 "function of type `{:s}'".format(
                                     str(backend), function.__class__.__name__))
        else:
            # We can only auto-detect theano and tensorflow since autograd and
            # pytorch functions are regular callables. In case we get passed a
            # callable without the 'backend' attribute set as checked above,
            # we therefore default to the canonical backend.
            if callable(function):
                backend = CallableBackend()
            else:
                for backend in [TheanoBackend(), TensorflowBackend()]:
                    if (backend.is_available() and
                            backend.is_compatible(function, argument)):
                        break
        if backend is None:
            # TODO: Add a static .name attribute to backend classes instead so
            #       we don't have to instantiate every backend here.
            backend_names = [str(Backend()) for Backend in _BACKENDS]
            raise ValueError(
                    "Cannot determine autodiff backend from cost function of "
                    "type `{:s}`. Available backends are: {:s}".format(
                        function.__class__.__name__, ", ".join(backend_names)))
        return backend

    def _compile(self):
        assert self._backend is not None
        self._compiled_function = self._backend.compile_function(
            self._function, self._arg)

    def _perform_differentiation(self, attr):
        assert self._backend is not None
        if isinstance(self._backend, CallableBackend):
            raise ValueError("CallableBackend does not support automatic "
                             "differentiation")
        method = getattr(self._backend, attr)
        derivative = method(self._function, self._arg)
        # Whatever the backend, the result of a call to an automatic
        # differentation routine will be a regular callable which we cannot
        # autodiff anymore. We therefore don't bother passing along self._arg.
        return Function(derivative)

    def compute_gradient(self):
        return self._perform_differentiation("compute_gradient")

    def compute_hessian(self):
        return self._perform_differentiation("compute_hessian")

    def __call__(self, *args, **kwargs):
        assert self._compiled_function is not None
        return self._compiled_function(*args, **kwargs)
