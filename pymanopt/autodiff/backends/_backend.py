import abc
import functools

import attrs
import numpy as np


def fail_on_complex_input(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if any(map(np.iscomplexobj, args)) or any(
            map(np.iscomplexobj, kwargs.values())
        ):
            raise TypeError("Autodiff backend does not support complex input")
        return function(*args, **kwargs)

    return wrapper


@attrs.define
class Backend(metaclass=abc.ABCMeta):
    """Abstract base class defining the interface for autodiff backends.

    Args:
        name: The name of the backend.
    """

    _name: str

    def __str__(self):
        return self._name

    @staticmethod
    def _assert_backend_available(method):
        """Decorator verifying the availability of a backend.

        Args:
            method: The method of a class to decorate.

        Returns:
            callable: The wrapped method.

        Raises:
            RuntimeError: If the backend isn't available.
        """

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if not self.is_available():
                raise RuntimeError(f"Backend '{self}' is not available")
            return method(self, *args, **kwargs)

        return wrapper

    @abc.abstractstaticmethod
    def is_available():
        """Checks whether the backend is available or not.

        Returns:
            True if backend is available, False otherwise.
        """

    @abc.abstractmethod
    def prepare_function(self, function):
        """Prepares a callable to be used with the backend.

        Args:
            function: A callable.

        Returns:
            A Python callable accepting and a ``numpy.ndarray`` and returning a
            scalar.
        """

    @abc.abstractmethod
    def generate_gradient_operator(self, function, num_arguments):
        """Creates a function to compute gradients of a function.

        Args:
            function: A callable.
            num_arguments: The number of arguments that ``function`` expects.

        Returns:
            A Python callable of the gradient of `function` accepting arguments
            according to the signature defined by `arguments`.
        """

    @abc.abstractmethod
    def generate_hessian_operator(self, function, num_arguments):
        """Creates a function to compute Hessian-vector products of a function.

        Args:
            function: A callable.
            num_arguments: The number of arguments that ``function`` expects.

        Returns:
            A Python callable evaluating the Hessian-vector product of
            ``function`` accepting arguments according to the signature defined
            by ``arguments``.
            The returned callable accepts a point of evaluation as a sequence
            of length ``num_arguments``, as well as a vector of the same shape
            that is right-multiplied to the Hessian.
        """
