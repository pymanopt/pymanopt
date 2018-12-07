from ._autograd import AutogradBackend
from ._callable import CallableBackend
from ._pytorch import PyTorchBackend
from ._tensorflow import TensorflowBackend
from ._theano import TheanoBackend

__all__ = ["AutogradBackend", "CallableBackend", "PyTorchBackend",
           "TensorflowBackend", "TheanoBackend"]


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

    def compute_gradient(self):
        assert self._backend is not None
        return self._backend.compute_gradient(self._function, self._arg)

    def compute_hessian(self):
        assert self._backend is not None
        return self._backend.compute_hessian(self._function, self._arg)

    def __call__(self, *args, **kwargs):
        assert self._compiled_function is not None
        return self._compiled_function(*args, **kwargs)
