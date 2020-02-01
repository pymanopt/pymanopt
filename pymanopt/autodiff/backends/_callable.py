from ._backend import Backend
from .. import make_tracing_backend_decorator


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
        return function

    def _raise_not_implemented_error(self, function, arguments):
        raise NotImplementedError(
            "No autodiff support available for the canonical '{}' "
            "backend".format(self))

    compute_gradient = _raise_not_implemented_error
    compute_hessian_vector_product = _raise_not_implemented_error


Callable = make_tracing_backend_decorator(_CallableBackend)
