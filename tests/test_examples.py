from examples import (
    closest_unit_norm_column_approximation,
    dominant_eigenvector,
    dominant_invariant_subspace,
    low_rank_matrix_approximation,
    low_rank_psd_matrix_approximation,
    multiple_linear_regression,
    optimal_rotations,
    packing_on_the_sphere,
    pca,
    rank_k_correlation_matrix_approximation
)
from nose2.tools import params

from ._test import TestCase


class TestExamples(TestCase):
    @params(*closest_unit_norm_column_approximation.SUPPORTED_BACKENDS)
    def test_closest_unit_norm_column_approximation(self, backend):
        closest_unit_norm_column_approximation.run(backend)

    @params(*dominant_eigenvector.SUPPORTED_BACKENDS)
    def test_dominant_eigenvector(self, backend):
        dominant_eigenvector.run(backend)

    @params(*dominant_invariant_subspace.SUPPORTED_BACKENDS)
    def test_dominant_invariant_subspace(self, backend):
        dominant_invariant_subspace.run(backend)

    @params(*low_rank_matrix_approximation.SUPPORTED_BACKENDS)
    def test_low_rank_matrix_approximation(self, backend):
        low_rank_matrix_approximation.run(backend)

    @params(*low_rank_psd_matrix_approximation.SUPPORTED_BACKENDS)
    def test_low_rank_psd_matrix_approximation(self, backend):
        low_rank_psd_matrix_approximation.run(backend)

    @params(*multiple_linear_regression.SUPPORTED_BACKENDS)
    def test_multiple_linear_regression(self, backend):
        multiple_linear_regression.run(backend)

    @params(*optimal_rotations.SUPPORTED_BACKENDS)
    def test_optimal_rotations(self, backend):
        optimal_rotations.run(backend)

    @params(*packing_on_the_sphere.SUPPORTED_BACKENDS)
    def test_packing_on_the_sphere(self, backend):
        packing_on_the_sphere.run(backend)

    @params(*pca.SUPPORTED_BACKENDS)
    def test_pca(self, backend):
        pca.run(backend)

    @params(*rank_k_correlation_matrix_approximation.SUPPORTED_BACKENDS)
    def test_rank_k_correlation_matrix_approximation(self, backend):
        rank_k_correlation_matrix_approximation.run(backend)
