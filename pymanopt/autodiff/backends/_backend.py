from functools import wraps


def assert_backend_available(f):
    @wraps(f)
    def inner(backend, *args, **kwargs):
        if not backend.is_available():
            raise RuntimeError(
                "Backend `{:s}` is not available".format(str(backend)))
        return f(backend, *args, **kwargs)
    return inner


class Backend:
    def __str__(self):
        return "<backend>"

    def __not_implemented(self, objective, argument):
        raise NotImplementedError

    compile_function = compute_gradient = compute_hessian = __not_implemented

    def __false(*args, **kwargs):
        return False

    is_available = is_compatible = __false
