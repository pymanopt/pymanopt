import unittest

from pymanopt.tools import flatten_arguments


class TestArgumentFlattening(unittest.TestCase):
    def _test_flatten_arguments(
            self, arguments, correctly_flattened_arguments):
        flattened_arguments = flatten_arguments(arguments)
        self.assertEqual(flattened_arguments, correctly_flattened_arguments)

        flattened_arguments_with_signature_hint = flatten_arguments(
            arguments, signature=arguments)
        self.assertEqual(flattened_arguments_with_signature_hint,
                         correctly_flattened_arguments)

    def test_single_argument(self):
        arguments = ("x",)
        self._test_flatten_arguments(arguments, arguments)

    def test_multiple_arguments(self):
        arguments = ("x", "y", "z")
        self._test_flatten_arguments(arguments, arguments)

    def test_nested_arguments(self):
        arguments = (("x", "y"), "z")
        self._test_flatten_arguments(arguments, ("x", "y", "z"))
