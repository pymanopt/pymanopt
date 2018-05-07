from ._callable import CallableBackend
from ._autograd import AutogradBackend
from ._pytorch import PyTorchBackend
from ._theano import TheanoBackend
from ._tensorflow import TensorflowBackend

_BACKENDS = (CallableBackend, AutogradBackend, PyTorchBackend, TheanoBackend,
             TensorflowBackend)
__all__ = [Backend.__name__ for Backend in _BACKENDS]


__all__ = ["TheanoBackend", "AutogradBackend", "TensorflowBackend"]
