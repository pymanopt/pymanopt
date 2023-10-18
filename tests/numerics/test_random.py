import numpy as np
import pytest

import pymanopt.numerics as nx


@pytest.mark.parametrize(
    "argument",
    [
        2,
        [2],
        np.array([2, 1]),
    ],
)
def test_normal(argument):
    np.random.seed(0)
    output = nx.random.normal(argument)
    np.random.seed(0)
    expected_output = np.random.normal(size=argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument",
    [
        [2],
        [2, 3],
    ],
)
def test_randn(argument):
    np.random.seed(0)
    output = nx.random.randn(*argument)
    np.random.seed(0)
    expected_output = np.random.randn(*argument)
    assert nx.allclose(output, expected_output)


@pytest.mark.parametrize(
    "argument",
    [
        (1, 3, 2),
        (1, 3, [2]),
        (1.1, 3.2, np.array([2, 1])),
    ],
)
def test_uniform(argument):
    np.random.seed(0)
    output = nx.random.uniform(*argument)
    np.random.seed(0)
    expected_output = np.random.uniform(*argument)
    assert nx.allclose(output, expected_output)
