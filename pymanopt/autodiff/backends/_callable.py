from ._backend import Backend
from .. import make_tracing_backend_decorator
from ...tools import flatten_arguments, unpack_arguments


class _CallableBackend(Backend):
    def __init__(self):
        super().__init__("Callable")

    @staticmethod
    def is_available():
        return True

    @Backend._assert_backend_available
    def is_compatible(self, function, arguments):
        return callable(function)

    @Backend._assert_backend_available
    def compile_function(self, function, arguments):
        flattened_arguments = flatten_arguments(arguments)
        if len(flattened_arguments) == 1:
            return function
        return unpack_arguments(function)

    def _raise_not_implemented_error(self, function, arguments):
        raise NotImplementedError(
            "No autodiff support available for the canonical '{}' "
            "backend".format(self))

    compute_gradient = compute_hessian = _raise_not_implemented_error


Callable = make_tracing_backend_decorator(_CallableBackend)
