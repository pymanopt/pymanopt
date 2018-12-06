from ._backend import Backend, assert_backend_available


class CallableBackend(Backend):
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

    @assert_backend_available
    def __not_implemented(self, objective, argument):
        raise NotImplementedError("No autodiff support available for the "
                                  "canonical callable backend")

    compute_gradient = compute_hessian = __not_implemented
