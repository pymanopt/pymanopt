import importlib
from typing import TYPE_CHECKING, Any, Callable

from .backend import Backend, DummyBackendSingleton  # noqa: F401


__all__ = ["Backend", "DummyBackendSingleton"]


def attach(
    package_name: str, submodules: list[str]
) -> tuple[Callable[[str], Any], Callable[[], list[str]], list[str]]:
    """Lazily loads submodules of a package."""
    __all__: list[str] = list(submodules)

    def __getattr__(name: str) -> Any:
        if name in submodules:
            return importlib.import_module(f"{package_name}.{name}")
        raise AttributeError(
            f"module '{package_name}' has no attribute '{name}"
        )

    def __dir__() -> list[str]:
        return __all__

    return __getattr__, __dir__, __all__


if TYPE_CHECKING:
    # This variable is only set to true in LSPs and type checkers and is
    # always false at runtime.
    from . import autograd_backend  # noqa: F401
    from . import jax_backend  # noqa: F401
    from . import numpy_backend  # noqa: F401
    from . import pytorch_backend  # noqa: F401
    from . import tensorflow_backend  # noqa: F401
else:
    # Lazily load the backend implementation modules.
    __getattr__, __dir__, __all__ = attach(
        __name__,
        [
            "numpy_backend",
            "jax_backend",
            "autograd_backend",
            "pytorch_backend",
            "tensorflow_backend",
        ],
    )

del attach, TYPE_CHECKING  # avoids using these inadvertently
