from ._callable import CallableBackend
from ._autograd import AutogradBackend
from ._pytorch import PyTorchBackend
from ._theano import TheanoBackend
from ._tensorflow import TensorflowBackend


class Function(object):
    def __init__(self, function, arg, backend):
        self._function = function
        self._arg = arg
        self._backend = backend

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
        self._compiled_function = self._backend.compile_function(
            self._function, self._arg)

    def _perform_differentiation(self, attr):
        assert self._backend is not None
        method = getattr(self._backend, attr)
        return method(self._function, self._arg)

    def compute_gradient(self):
        return self._perform_differentiation("compute_gradient")

    def compute_hessian(self):
        return self._perform_differentiation("compute_hessian")

    def __call__(self, *args, **kwargs):
        assert self._compiled_function is not None
        return self._compiled_function(*args, **kwargs)
