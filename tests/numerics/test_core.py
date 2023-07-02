import numpy as np
import pytest

import pymanopt.numerics as nx


@pytest.mark.parametrize(
    "argument, expected_output", [(np.array([-4, 2]), np.array([4, 2]))]
)
def test_abs(argument, expected_output):
    output = nx.abs(argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument, expected_output", [(np.array([-4, 2]), np.exp([-4, 2]))]
)
def test_exp(argument, expected_output):
    output = nx.exp(argument)
    assert nx.allclose(output, expected_output)
