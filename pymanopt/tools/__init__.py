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


# TODO(nkoep): rename this to flatten_arguments
def flatten_args(args):
    """
    Takes a sequence of arguments containing tuples/lists of arguments or
    unary arguments and returns a flattened tuple of arguments, e.g.
    `flatten_args((1, 2), 3)' produces the tuple `(1, 2, 3)'.
    """
    flattened_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            flattened_args.extend(arg)
        else:
            flattened_args.append(arg)
    return tuple(flattened_args)


def unpack_arguments(f):
    """
    Wraps a function accepting a single sequence of arguments and calls the
    function with unpacked arguments.
    """
    @functools.wraps(f)
    def inner(args):
        return f(*flatten_args(args))
    return inner


def group_return_values(f, args):
    if len(args) == 1:
        @functools.wraps(f)
        def inner(*args):
            return f(*args)
        return inner

    signature = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            signature.append(len(arg))
        else:
            signature.append(1)

    @functools.wraps(f)
    def inner(*args):
        # TODO(nkoep): The function might be hot. Come up with a more elegant
        #              implementation.
        returns = f(*args)
        groups = []
        i = 0
        for n in signature:
            if n == 1:
                groups.append(returns[i])
            else:
                groups.append(returns[i : i + n])
            i += n
        return groups
    return inner
