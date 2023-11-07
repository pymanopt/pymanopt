import pytest
import scipy

import pymanopt.numerics as nx

from tests.numerics import _test_numerics_supported_backends


@pytest.mark.parametrize(
    "argument_a, argument_b, expected_output",
    [
        (2, 0, scipy.special.comb(2, 0)),
        (3, 2, scipy.special.comb(3, 2)),
    ],
)
@_test_numerics_supported_backends()
def test_comb(argument_a, argument_b, expected_output):
    actual = nx.special.comb(argument_a, argument_b)
    assert nx.allclose(actual, expected_output)
