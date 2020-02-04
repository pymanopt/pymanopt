import abc
import functools


class Backend(metaclass=abc.ABCMeta):
    """Abstract base class defining the interface autodiff backends must
    implement.

    Parameters
    ----------
    name : str
        The name of the backend.
    """

    def __init__(self, name):
        self._name = name

    def __str__(self):
        return self._name

    def _assert_backend_available(method):
        """Decorator which verifies the availability of a backend before
        evaluating the decorated function, raising a RuntimeError exception if
        the backend isn't available.

        Parameters
        ----------
        method : callable
            The method of a class to decorate.

        Returns
        -------
        callable
            The wrapped method.
        """
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if not self.is_available():
                raise RuntimeError(
                    "Backend '{}' is not available".format(self))
            return method(self, *args, **kwargs)
        return wrapper

    @abc.abstractstaticmethod
    def is_available():
        """Checks whether the backend is available or not.

        Returns
        -------
        bool
            True if backend is available, False otherwise.
        """

    @abc.abstractmethod
    def is_compatible(self, function, arguments):
        """Checks whether `function` and `arguments` are compatible with the
        backend.

        Parameters
        ----------
        function
            Python callable or a backend-specific computational graph node.
        arguments
            A backend-dependent representation of the arguments `function`
            expects.

        Returns
        -------
        bool
            True if the backend is compatible with `function` and `arguments`,
            False otherwise.
        """

    @abc.abstractmethod
    def compile_function(self, function, arguments):
        """Compiles a function into a Python callable.

        Parameters
        ----------
        function
            Python callable or a backend-specific computational graph node.
        arguments
            A backend-dependent representation of the arguments `function`
            expects.

        Returns
        -------
        compiled_function : callable
            A Python callable accepting arguments according to the signature
            defined by `arguments`.
        """

    @abc.abstractmethod
    def compute_gradient(self, function, arguments):
        """Computes the gradient of a function and turns it into a Python
        callable.

        Parameters
        ----------
        function
            Python callable or a backend-specific computational graph node.
        arguments
            A backend-dependent representation of the arguments `function`
            expects.

        Returns
        -------
        gradient : callable
            A Python callable of the gradient of `function` accepting arguments
            according to the signature defined by `arguments`.
        """

    @abc.abstractmethod
    def compute_hessian_vector_product(self, function, arguments):
        """Computes the Hessian-vector product of function a function and turns
        it into a Python callable.

        Parameters
        ----------
        function
            Python callable or a backend-specific computational graph node.
        arguments
            A backend-dependent representation of the arguments `function`
            expects.

        Returns
        -------
        hessian_vector_product : callable
            A Python callable evaluating the Hessian-vector product of
            `function` accepting arguments according to the signature defined
            by `arguments`. The returned callable accepts a point of evaluation
            according to `arguments`, as well as a vector that is
            right-multiplied to the Hessian.
        """
