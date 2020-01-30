from examples import (
    closest_unit_norm_column_approximation,
    dominant_eigenvector,
    dominant_invariant_subspace,
    low_rank_matrix_approximation,
    multiple_linear_regression
)
from nose2.tools import params

from ._test import TestCase


class TestExamples(TestCase):
    @params(*closest_unit_norm_column_approximation.SUPPORTED_BACKENDS)
    def test_closest_unit_norm_column_approximation(self, backend):
        closest_unit_norm_column_approximation.run(backend=backend)

    @params(*dominant_eigenvector.SUPPORTED_BACKENDS)
    def test_dominant_eigenvector(self, backend):
        dominant_eigenvector.run(backend=backend)

    @params(*dominant_invariant_subspace.SUPPORTED_BACKENDS)
    def test_dominant_invariant_subspace(self, backend):
        dominant_invariant_subspace.run(backend=backend)

    @params(*low_rank_matrix_approximation.SUPPORTED_BACKENDS)
    def test_low_rank_matrix_approximation(self, backend):
        low_rank_matrix_approximation.run(backend=backend)

    @params(*multiple_linear_regression.SUPPORTED_BACKENDS)
    def test_multiple_linear_regression(self, backend):
        multiple_linear_regression.run(backend=backend)
