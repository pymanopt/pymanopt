from functools import wraps
from typing import Union


class SequenceDispatch:
    """A custom dispatcher for functions that take a Sequence of data as input.

    This class is initialized with a default function that will be called if no
    specific function is registered for the type of the first element.

    Functions can be registered to specific types using the `.register` method
    decorated with the type they should handle.

    Example:
    --------
    ```
    from typing import Sequence

    @SequenceDispatch
    def dispatcher(_):
        pass

    @dispatcher.register
    def handle_int(data_list: Sequence[int]):
        print('Handle a list of integers')

    @dispatcher.register
    def handle_str(data_list: Sequence[str]):
        print('Handle a list of strings')

    data = [1, 2, 3]
    dispatcher(data)  # Prints 'Handle a list of integers'
    data = ['a', 'b', 'c']
    dispatcher(data)  # Prints 'Handle a list of strings'
    ```
    """
    def __init__(self, func):
        func = wraps(func)(self)
        self.default_func = func
        self.registry = {}

    def register(self, func):
        # Get the type annotations
        arg_types = func.__annotations__

        # Get list of all keys in arg_types that are not 'return'
        arg_types_keys = [k for k in arg_types.keys() if k != 'return']

        # Extract the type from Sequence[type]
        type_ = arg_types[arg_types_keys[0]].__args__[0]

        if type_:
            self.registry[type_] = func

        return func

    def __call__(self, *args, **kwargs):
        data_list = args[0]
        if not data_list:  # Handle the case of an empty list
            return self.default_func(*args, **kwargs)

        d = data_list[0]
        while isinstance(d , (tuple, list)):
            d = d[0]
        type_ = type(d)
        func = self.registry.get(type_, self.default_func)
        return func(*args, **kwargs)
