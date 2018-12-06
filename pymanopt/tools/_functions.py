from .autodiff import (Function, AutogradBackend, CallableBackend,
                       PyTorchBackend, TensorflowBackend, TheanoBackend)


def _make_wrapper_class(name, Backend):
    class _Function(Function):
        def __init__(self, function, arg=None):
            super(_Function, self).__init__(function, arg, Backend())
    _Function.__name__ = name
    return _Function


AutogradFunction = _make_wrapper_class("AutogradFunction", AutogradBackend)
CallableFunction = _make_wrapper_class("CallableFunction", CallableBackend)
PyTorchFunction = _make_wrapper_class("PyTorchFunction", PyTorchBackend)
TensorflowFunction = _make_wrapper_class("TensorflowFunction",
                                         TensorflowBackend)
TheanoFunction = _make_wrapper_class("TheanoFunction", TheanoBackend)
