__all__ = ["autograd", "jax", "numpy", "pytorch", "tensorflow"]

from .. import backend_decorator_factory
from ._autograd import AutogradBackend
from ._jax import JaxBackend
from ._numpy import NumPyBackend
from ._pytorch import PyTorchBackend
from ._tensorflow import TensorFlowBackend


autograd = backend_decorator_factory(AutogradBackend)
jax = backend_decorator_factory(JaxBackend)
numpy = backend_decorator_factory(NumPyBackend)
pytorch = backend_decorator_factory(PyTorchBackend)
tensorflow = backend_decorator_factory(TensorFlowBackend)
