import importlib

from .array_t import (
    AVAILABLE_NUMERICS_BACKENDS,
    SUPPORTED_NUMERICS_BACKENDS,
    array_t,
)
from .core import DummyNumericsBackend, NumericsBackend


__all__ = [
    "SUPPORTED_NUMERICS_BACKENDS",
    "AVAILABLE_NUMERICS_BACKENDS",
    "array_t",
    "NumericsBackend",
    "DummyNumericsBackend",
]

for backend in AVAILABLE_NUMERICS_BACKENDS:
    module = importlib.import_module(f"._{backend}", package=__name__)
    globals().update(
        {k: v for k, v in module.__dict__.items() if not k.startswith("_")}
    )
