import collections
import functools


def make_enum(name, fields):
    return collections.namedtuple(name, fields)(*range(len(fields)))


class ndarraySequenceMixin:
    """Mixin to ensure that operations on sequences of numpy.ndarrays with
    scalar numpy data types such as numpy.float64 don't attempt to vectorize
    the scalar variable.

    Notes
    -----
    Refer to [1]_ and [2]_ for details.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/numpy/reference/arrays.classes.html
    .. [2] https://github.com/pymanopt/pymanopt/issues/49
    """
    __array_priority__ = 1000
    __array_ufunc__ = None  # Available since numpy 1.13


def unpack_singleton_sequence_return_value(function):
    """Function decorator which unwraps the return value of ``function`` if it
    is a sequence containing only a single element.
    """
    @functools.wraps(function)
    def wrapper(*args):
        result = function(*args)
        if not hasattr(result, "__iter__") or len(result) != 1:
            raise ValueError("Function did not return a singleton sequence")
        return result[0]
    return wrapper


def bisect_sequence(sequence):
    """Splits a sequence of even length into two equal-length subsequences.
    """
    assert hasattr(sequence, "__iter__")
    num_items = len(sequence)
    if num_items % 2 == 1:
        raise ValueError("Sequence must have an even number of elements")
    return sequence[:num_items // 2], sequence[num_items // 2:]
