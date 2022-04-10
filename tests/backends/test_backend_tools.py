import unittest

from pymanopt.tools import (
    bisect_sequence,
    unpack_singleton_sequence_return_value,
)


class TestArgumentFlattening(unittest.TestCase):
    def test_bisect_sequence(self):
        sequence = range(10)
        half1, half2 = bisect_sequence(sequence)
        self.assertEqual(half1, range(5))
        self.assertEqual(half2, range(5, 10))
        with self.assertRaises(ValueError):
            bisect_sequence(range(11))

    def test_unpack_singleton_sequence_return_value(self):
        @unpack_singleton_sequence_return_value
        def f():
            return (1,)

        self.assertEqual(f(), 1)

        @unpack_singleton_sequence_return_value
        def g():
            return (1, 2)

        with self.assertRaises(ValueError):
            g()

        @unpack_singleton_sequence_return_value
        def h():
            return None

        with self.assertRaises(ValueError):
            h()
