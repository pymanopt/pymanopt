import functools
import typing


class ndarraySequenceMixin:
    # The following attributes ensure that operations on sequences of
    # np.ndarrays with scalar numpy data types such as np.float64 don't attempt
    # to vectorize the scalar variable. Refer to
    #
    #     https://docs.scipy.org/doc/numpy/reference/arrays.classes.html
    #     https://github.com/pymanopt/pymanopt/issues/49
    #
    # for details.
    __array_priority__ = 1000
    __array_ufunc__ = None


def return_as_class_instance(method=None, *, unpack=True):
    """Method decorator to wrap return values in a class instance."""

    def make_wrapper(function):
        @functools.wraps(function)
        def wrapper(self, *args, **kwargs):
            return_value = function(self, *args, **kwargs)
            if unpack:
                return self.__class__(*return_value)
            return self.__class__(return_value)

        return wrapper

    if method is not None and callable(method):
        return make_wrapper(method)
    return make_wrapper


def unpack_singleton_sequence_return_value(function):
    """Decorator to unwrap singleton return values.

    Function decorator which unwraps the return value of ``function`` if it
    is a sequence containing only a single element.
    """

    @functools.wraps(function)
    def wrapper(*args):
        result = function(*args)
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise ValueError("Function did not return a singleton sequence")
        return result[0]

    return wrapper


def bisect_sequence(sequence):
    num_items = len(sequence)
    if num_items % 2 == 1:
        raise ValueError("Sequence must have an even number of elements")
    return sequence[: num_items // 2], sequence[num_items // 2 :]


def is_sequence(instance):
    return not isinstance(instance, str) and isinstance(
        instance, typing.Sequence
    )


def extend_docstring(text: str):
    def inner(cls):
        cls.__doc__ += text
        return cls

    return inner
