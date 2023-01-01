from ._backend import Backend


def _raise_not_implemented_error(*args, **kwargs):
    raise NotImplementedError(
        "No autodiff support available for the NumPy backend"
    )


class NumPyBackend(Backend):
    def __init__(self):
        super().__init__("NumPy")

    @staticmethod
    def is_available():
        return True

    @Backend._assert_backend_available
    def prepare_function(self, function):
        return function

    generate_gradient_operator = _raise_not_implemented_error
    generate_hessian_operator = _raise_not_implemented_error
