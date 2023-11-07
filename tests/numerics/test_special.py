import pytest
import scipy

import pymanopt.numerics as nx


@pytest.mark.parametrize(
    "argument_a, argument_b, expected_output",
    [
        (2, 0, scipy.special.comb(2, 0)),
        (3, 2, scipy.special.comb(3, 2)),
    ],
)
def test_comb(argument_a, argument_b, expected_output):
    actual = nx.special.comb(argument_a, argument_b)
    assert nx.allclose(actual, expected_output)
