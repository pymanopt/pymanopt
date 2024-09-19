import functools


try:
    import jax
except ImportError:
    jax = None
else:
    import jax.numpy as jnp

    # for backward compatibility with older versions of jax
    try:
        from jax import config
    except ImportError:
        from jax.config import config

    config.update("jax_enable_x64", True)

from ...tools import bisect_sequence, unpack_singleton_sequence_return_value
from ._backend import Backend


def conjugate_result(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return list(map(jnp.conj, function(*args, **kwargs)))

    return wrapper


class JaxBackend(Backend):
    def __init__(self):
        super().__init__("Jax")

    @staticmethod
    def is_available():
        return jax is not None

    @Backend._assert_backend_available
    def generate_gradient_operator(self, function, num_arguments):
        gradient = conjugate_result(
            jax.grad(function, argnums=range(num_arguments)))
        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(gradient)
        return gradient

    @Backend._assert_backend_available
    def generate_hessian_operator(self, function, num_arguments):
        @conjugate_result
        def hessian_vector_product(arguments, vectors):
            return jax.jvp(
                jax.grad(function, argnums=range(num_arguments)),
                arguments,
                vectors,
            )[1]

        @functools.wraps(hessian_vector_product)
        def wrapper(*args):
            arguments, vectors = bisect_sequence(args)
            return hessian_vector_product(arguments, vectors)

        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(wrapper)
        return wrapper
