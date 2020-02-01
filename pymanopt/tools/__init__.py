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


def flatten_arguments(arguments, signature=None):
    """Takes a sequence `arguments` containing tuples/lists of arguments or
    unary arguments and returns a flattened tuple of arguments, e.g.
    `flatten_arguments([1, 2], 3)` produces the tuple `(1, 2, 3)`. If the
    nesting cannot be inferred from the types of objects contained in
    `arguments` itself, one may pass the optional argument `signature` instead.
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
        return function(*flatten_arguments(arguments, signature=signature))
    return inner


def group_return_values(function, signature):
    """Returns a wrapped version of `function` which groups the return values
    of the function in the same way as defined by the signature given by
    `signature`.
    """
    if len(signature) == 1:
        @functools.wraps(function)
        def inner(*args):
            return function(*args)
        return inner

    group_sizes = []
    for element in signature:
        if isinstance(element, (list, tuple)):
            group_sizes.append(len(element))
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


def unpack_singleton_iterable_return_value(function):
    """Function decorator which unwraps^
    """
    @functools.wraps(function)
    def wrapper(*args):
        result = function(*args)
        assert isinstance(result, (list, tuple))
        return result[0]
    return wrapper
