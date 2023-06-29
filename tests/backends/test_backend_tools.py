import pytest

from pymanopt.tools import (
    bisect_sequence,
    unpack_singleton_sequence_return_value,
)


class TestArgumentFlattening:
    def test_bisect_sequence(self):
        sequence = range(10)
        half1, half2 = bisect_sequence(sequence)
        assert half1 == range(5)
        assert half2 == range(5, 10)
        with pytest.raises(ValueError):
            bisect_sequence(range(11))

    def test_unpack_singleton_sequence_return_value(self):
        @unpack_singleton_sequence_return_value
        def f():
            return (1,)

        assert f() == 1

        @unpack_singleton_sequence_return_value
        def g():
            return (1, 2)

        with pytest.raises(ValueError):
            g()

        @unpack_singleton_sequence_return_value
        def h():
            return None

        with pytest.raises(ValueError):
            h()
