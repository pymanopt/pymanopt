__all__ = ["jax", "numpy", "pytorch", "tensorflow"]

from .. import backend_decorator_factory
from ._jax import JaxBackend
from ._numpy import NumPyBackend
from ._pytorch import PyTorchBackend
from ._tensorflow import TensorFlowBackend


jax = backend_decorator_factory(JaxBackend)
numpy = backend_decorator_factory(NumPyBackend)
pytorch = backend_decorator_factory(PyTorchBackend)
tensorflow = backend_decorator_factory(TensorFlowBackend)
