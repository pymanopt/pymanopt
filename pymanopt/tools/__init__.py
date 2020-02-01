import collections
import functools


def make_enum(name, fields):
    return collections.namedtuple(name, fields)(*range(len(fields)))


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
    __array_ufunc__ = None  # Available since numpy 1.13


def unpack_singleton_sequence_return_value(function):
    """Function decorator which unwraps the return value of ``function`` if it
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
    return sequence[:num_items // 2], sequence[num_items // 2:]
