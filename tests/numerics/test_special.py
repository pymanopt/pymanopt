import pytest
import scipy

import pymanopt.numerics as nx


@pytest.mark.parametrize(
    "argument_a, argument_b",
    [
        (2, 0),
        (3, 2)
    ],
)
def test_comb(argument_a, argument_b):
    actual = nx.special.comb(argument_a, argument_b)
    expected = scipy.special.comb(argument_a, argument_b)
    assert nx.allclose(actual, expected)
