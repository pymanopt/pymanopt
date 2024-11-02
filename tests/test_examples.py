import pytest

from examples import (
    closest_unit_norm_column_approximation,
    dominant_eigenvector,
    dominant_invariant_complex_subspace,
    dominant_invariant_subspace,
    low_rank_matrix_approximation,
    low_rank_psd_matrix_approximation,
    multiple_linear_regression,
    optimal_rotations,
    packing_on_the_sphere,
    pca,
    rank_k_correlation_matrix_approximation,
)
from examples.advanced import check_gradient, check_hessian, check_retraction


SUPPORTED_BACKENDS = {"numpy", "autograd", "jax", "pytorch", "tensorflow"}


class TestExamples:
    @pytest.mark.parametrize(
        "backend",
        list(
            set(
                closest_unit_norm_column_approximation.SUPPORTED_BACKENDS
            ).intersection(SUPPORTED_BACKENDS)
        ),
    )
    def test_closest_unit_norm_column_approximation(self, backend):
        closest_unit_norm_column_approximation.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(dominant_eigenvector.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_dominant_eigenvector(self, backend):
        dominant_eigenvector.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(dominant_invariant_subspace.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_dominant_invariant_subspace(self, backend):
        dominant_invariant_subspace.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(dominant_invariant_subspace.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_dominant_invariant_complex_subspace(self, backend):
        dominant_invariant_complex_subspace.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(low_rank_matrix_approximation.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_low_rank_matrix_approximation(self, backend):
        low_rank_matrix_approximation.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(
                low_rank_psd_matrix_approximation.SUPPORTED_BACKENDS
            ).intersection(SUPPORTED_BACKENDS)
        ),
    )
    def test_low_rank_psd_matrix_approximation(self, backend):
        low_rank_psd_matrix_approximation.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(multiple_linear_regression.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_multiple_linear_regression(self, backend):
        multiple_linear_regression.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(optimal_rotations.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_optimal_rotations(self, backend):
        optimal_rotations.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(packing_on_the_sphere.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_packing_on_the_sphere(self, backend):
        packing_on_the_sphere.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(set(pca.SUPPORTED_BACKENDS).intersection(SUPPORTED_BACKENDS)),
    )
    def test_pca(self, backend):
        pca.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(
                rank_k_correlation_matrix_approximation.SUPPORTED_BACKENDS
            ).intersection(SUPPORTED_BACKENDS)
        ),
    )
    def test_rank_k_correlation_matrix_approximation(self, backend):
        rank_k_correlation_matrix_approximation.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(check_gradient.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_check_gradient(self, backend):
        check_gradient.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(check_gradient.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_check_hessian(self, backend):
        check_hessian.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(check_retraction.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_check_retraction(self, backend):
        check_retraction.run(backend)
