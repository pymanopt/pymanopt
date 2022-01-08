__all__ = ["autograd", "numpy", "pytorch", "tensorflow"]

from .. import make_tracing_backend_decorator
from ._autograd import _AutogradBackend
from ._numpy import _NumPyBackend
from ._pytorch import _PyTorchBackend
from ._tensorflow import _TensorFlowBackend


autograd = make_tracing_backend_decorator(_AutogradBackend)
numpy = make_tracing_backend_decorator(_NumPyBackend)
pytorch = make_tracing_backend_decorator(_PyTorchBackend)
tensorflow = make_tracing_backend_decorator(_TensorFlowBackend)
