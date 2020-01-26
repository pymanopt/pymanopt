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


def _flatten_arguments_from_signature(arguments, signature):
    flattened_arguments = []
    for i, group in enumerate(signature):
        if isinstance(group, (list, tuple)):
            flattened_arguments.extend(arguments[i])
        else:
            flattened_arguments.append(arguments[i])
    return tuple(flattened_arguments)


# TODO(nkoep): rename this to flatten_arguments
def flatten_args(arguments, signature=None):
    """Takes a sequence of arguments containing tuples/lists of arguments or
    unary arguments and returns a flattened tuple of arguments, e.g.
    `flatten_args([1, 2], 3)' produces the tuple `(1, 2, 3)'.
    """
    if signature is not None:
        return _flatten_arguments_from_signature(arguments, signature)

    flattened_arguments = []
    for argument in arguments:
        if isinstance(argument, (list, tuple)):
            flattened_arguments.extend(argument)
        else:
            flattened_arguments.append(argument)
    return tuple(flattened_arguments)


def unpack_arguments(function, signature=None):
    """A decorator which wraps a function accepting a single sequence of
    arguments and calls the function with unpacked arguments. If given, the
    call arguments are unpacked according to the `signature' which is a string
    representation of the argument grouping/nesting, e.g. `(("x", "y"), "z")'.
    """
    @functools.wraps(function)
    def inner(arguments):
        return function(*flatten_args(arguments, signature=signature))
    return inner


def group_return_values(function, arguments):
    """Returns a wrapped version of `function' which groups the return values
    of the function in the same way as defined by the signature defined by
    `arguments'.
    """
    if len(arguments) == 1:
        @functools.wraps(function)
        def inner(*args):
            return function(*args)
        return inner

    group_sizes = []
    for argument in arguments:
        if isinstance(argument, (list, tuple)):
            group_sizes.append(len(argument))
        else:
            group_sizes.append(1)

    @functools.wraps(function)
    def inner(*args):
        # TODO(nkoep): This function might be hot. Can we come up with a more
        #              elegant implementation?
        return_values = function(*args)
        groups = []
        i = 0
        for n in group_sizes:
            if n == 1:
                groups.append(return_values[i])
            else:
                groups.append(return_values[i:i+n])
            i += n
        return groups
    return inner
