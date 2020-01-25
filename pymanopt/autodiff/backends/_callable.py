from ._backend import Backend, assert_backend_available
from .. import make_tracing_backend_decorator


class _CallableBackend(Backend):
    def __str__(self):
        return "callable"

    @staticmethod
    def is_available():
        return True

    @assert_backend_available
    def is_compatible(self, objective, argument):
        return callable(objective)

    @assert_backend_available
    def compile_function(self, objective, argument):
        return objective

    def _raise_not_implemented_error(self, objective, argument):
        raise NotImplementedError("No autodiff support available for the "
                                  "canonical 'Callable' backend")

    compute_gradient = compute_hessian = _raise_not_implemented_error


Callable = make_tracing_backend_decorator(_CallableBackend)
