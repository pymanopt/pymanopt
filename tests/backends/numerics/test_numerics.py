import inspect
import warnings

import numpy as np
import pytest
import scipy

from pymanopt.backends import Backend


@pytest.mark.parametrize(
    "inp, expected_output",
    [
        ([-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]),
        ([1, 0, 1], [1, 0, 1]),
        ([-127, -4], [127, 4]),
    ],
)
def test_abs(inp, expected_output, any_backend: Backend):
    bk = any_backend
    bk.assert_allclose(bk.abs(bk.array(inp)), bk.array(expected_output))


@pytest.mark.parametrize(
    "inp, expected_output",
    [
        ([True, False, True], False),
        ([True, True, True], True),
        ([], True),
    ],
)
def test_all(inp, expected_output, any_backend: Backend):
    assert any_backend.all(inp) == expected_output


@pytest.mark.parametrize(
    "inp, expected_output",
    [
        ([False, False, False], False),
        ([True, False, True], True),
        ([True, True, False], True),
        ([], False),
    ],
)
def test_any(inp, expected_output, any_backend: Backend):
    assert any_backend.any(inp) == expected_output


@pytest.mark.parametrize(
    "inp, expected_output",
    [
        (2, [[1.0, 0.0], [0.0, 1.0]]),
        (3, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ],
)
def test_eye(inp, expected_output, any_backend: Backend):
    bk = any_backend
    output = bk.eye(inp)
    expected_output = bk.array(expected_output)
    assert output.dtype == expected_output.dtype
    bk.assert_allclose(output, expected_output)


def test_logm(real_backend: Backend):
    bk = real_backend
    # test for a single matrix
    m = 10
    a = bk.reshape(bk.random_uniform(m), (-1, 1))
    q, _ = bk.linalg_qr(bk.random_normal(size=(m, m)))
    # A is a positive definite matrix
    A = q @ (a * bk.conjugate_transpose(q))
    logmA = bk.array(scipy.linalg.logm(np.asarray(A)))
    bk.assert_allclose(
        bk.linalg_logm(A, positive_definite=True), logmA, atol=1e-3
    )

    # test for several stacked matrices
    k = 4
    m = 10
    A = []
    L = []
    for _ in range(k):
        a = bk.reshape(bk.random_uniform(size=m), (-1, 1))
        q, _ = bk.linalg_qr(bk.random_normal(size=(m, m)))
        A.append(q @ (a * bk.conjugate_transpose(q)))
        L.append(bk.array(scipy.linalg.logm(np.asarray(A[-1]))))
    A = bk.stack(A)
    L = bk.stack(L)
    bk.assert_allclose(bk.linalg_logm(A, positive_definite=True), L, atol=1e-3)


@pytest.mark.skip
def test_multilogm_complex_positive_definite(complex_backend: Backend):
    bk = complex_backend
    k = 4
    m = 10
    shape = (k, m, m)
    A = bk.random_normal(size=shape)
    A = A @ bk.conjugate_transpose(A)
    # Compare fast path for positive definite matrices vs. general slow one.
    bk.assert_allclose(
        bk.linalg_logm(A, positive_definite=True),
        bk.linalg_logm(A, positive_definite=False),
    )


def test_conj(real_backend: Backend):
    complex_backend = real_backend.to_complex_backend()
    x = real_backend.random_randn(2, 2)
    y = complex_backend.random_randn(2, 2)
    for arr in [x, y]:
        complex_backend.assert_allclose(
            real_backend.transpose(arr), complex_backend.transpose(arr)
        )
        complex_backend.assert_allclose(
            real_backend.conjugate_transpose(arr),
            complex_backend.conjugate_transpose(arr),
        )
        print(real_backend.transpose(arr).dtype)
        print(real_backend.conjugate_transpose(arr).dtype)
        print(complex_backend.transpose(arr).dtype)
        print(complex_backend.conjugate_transpose(arr).dtype)


def test_signatures_match_base(any_backend: Backend):
    def parameters(method):
        signature = inspect.signature(method)
        return [(p.name, p.default) for p in signature.parameters.values()]

    for name, method in vars(Backend).items():
        if name.startswith(("_", "generate_")) or not inspect.isfunction(
            method
        ):
            continue
        assert parameters(method) == parameters(
            getattr(type(any_backend), name)
        ), name


@pytest.mark.parametrize("axis1, axis2", [(0, 2), (-2, -1), (2, 1)])
def test_diagonal(axis1, axis2, any_backend: Backend):
    bk = any_backend
    x = np.arange(24.0).reshape(2, 3, 4)
    bk.assert_allclose(
        bk.diagonal(bk.array(x), axis1, axis2),
        bk.array(np.diagonal(x, axis1=axis1, axis2=axis2)),
    )


@pytest.mark.parametrize("descending", [False, True])
def test_sort(descending, real_backend: Backend):
    bk = real_backend
    x = np.array([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]])
    expected = -np.sort(-x) if descending else np.sort(x)
    bk.assert_allclose(
        bk.sort(bk.array(x), descending=descending), bk.array(expected)
    )


def test_expm_symmetric_complex(complex_backend: Backend):
    bk = complex_backend
    x = np.array([[1.0, 1j], [-1j, 2.0]])
    bk.assert_allclose(
        bk.linalg_expm(bk.array(x), symmetric=True),
        bk.array(scipy.linalg.expm(x)),
    )


def test_assert_equal(any_backend: Backend):
    bk = any_backend
    bk.assert_equal(bk.array([1.0, 2.0]), bk.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        bk.assert_equal(bk.array([1.0, 2.0]), bk.array([1.0, 3.0]))


def test_logical_not(any_backend: Backend):
    bk = any_backend
    result = bk.logical_not(bk.array([1.0, 0.0]) == 0)
    np.testing.assert_array_equal(np.asarray(result), [True, False])


def test_matrix_rank(real_backend: Backend):
    bk = real_backend
    x = np.stack([np.eye(3), np.diag([1.0, 1.0, 0.0])])
    np.testing.assert_array_equal(
        np.asarray(bk.linalg_matrix_rank(bk.array(x))), [3, 2]
    )
    assert bk.linalg_matrix_rank(bk.array(x[1])) == 2


@pytest.mark.parametrize("shape", [(3, 3), (2, 3, 3), (2, 4, 3, 3)])
def test_trace(shape, any_backend: Backend):
    bk = any_backend
    x = np.arange(np.prod(shape), dtype=float).reshape(shape)
    bk.assert_allclose(
        bk.trace(bk.array(x)), bk.array(np.trace(x, axis1=-2, axis2=-1))
    )


@pytest.mark.parametrize("method", ["random_normal", "random_uniform"])
def test_random_single_sample(method, any_backend: Backend):
    bk = any_backend
    sample = getattr(bk, method)()
    assert tuple(sample.shape) == ()
    assert sample.dtype == bk.zeros(()).dtype


def test_random_normal_scale(any_backend: Backend):
    bk = any_backend
    samples = np.asarray(bk.random_normal(loc=1.0, scale=10.0, size=10000))
    parts = (
        [samples.real, samples.imag] if np.iscomplexobj(samples) else [samples]
    )
    for part in parts:
        assert np.std(part) == pytest.approx(10.0, rel=0.05)
        assert np.mean(part) == pytest.approx(1.0, abs=0.5)


def test_argmin(real_backend: Backend):
    bk = real_backend
    x = np.array([[3.0, 0.0], [1.0, 2.0]])
    assert bk.argmin(bk.array(x)) == np.argmin(x)


@pytest.mark.parametrize(
    "x", [[1.0, 2.0], [[1.0, 2.0], [3.0, 4.0]]], ids=["vector", "matrix"]
)
def test_diag(x, any_backend: Backend):
    bk = any_backend
    bk.assert_allclose(bk.diag(bk.array(x)), bk.array(np.diag(x)))


def test_where(any_backend: Backend):
    bk = any_backend
    condition = np.array([[True, False], [False, True]])
    indices = bk.where(bk.array(condition) == 1)
    assert isinstance(indices, tuple)
    assert len(indices) == 2
    for index, expected in zip(indices, np.where(condition)):
        np.testing.assert_array_equal(np.asarray(index), expected)
    bk.assert_allclose(
        bk.where(bk.array(condition) == 1, bk.ones((2, 2)), bk.zeros((2, 2))),
        bk.array(condition),
    )


@pytest.mark.parametrize(
    "arrays",
    [
        [[1.0, 2.0], [3.0, 4.0]],
        [[[1.0], [2.0]], [[3.0], [4.0]]],
    ],
    ids=["vectors", "matrices"],
)
@pytest.mark.parametrize("stack", ["hstack", "vstack"])
def test_hstack_vstack(stack, arrays, any_backend: Backend):
    bk = any_backend
    bk.assert_allclose(
        getattr(bk, stack)([bk.array(a) for a in arrays]),
        bk.array(getattr(np, stack)(arrays)),
    )


@pytest.mark.parametrize(
    "shape, reps", [((2,), (2, 1)), ((1, 2), 2), ((2, 2), (3, 1, 1))]
)
def test_tile(shape, reps, any_backend: Backend):
    bk = any_backend
    x = np.arange(np.prod(shape), dtype=float).reshape(shape)
    bk.assert_allclose(bk.tile(bk.array(x), reps), bk.array(np.tile(x, reps)))


@pytest.mark.parametrize(
    "a_shape, x_shape", [((1, 1), (1,)), ((2, 3, 3), (3,)), ((3, 3), (3, 2))]
)
def test_matvec(a_shape, x_shape, any_backend: Backend):
    bk = any_backend
    a = np.arange(np.prod(a_shape), dtype=float).reshape(a_shape)
    x = np.arange(np.prod(x_shape), dtype=float).reshape(x_shape)
    result = bk.matvec(bk.array(a), bk.array(x))
    assert tuple(result.shape) == (a @ x).shape
    bk.assert_allclose(result, bk.array(a @ x))


def test_allclose(any_backend: Backend):
    bk = any_backend
    ones = bk.ones((2, 3))
    assert bk.allclose(ones, bk.ones((3,))) is True
    assert bk.allclose(bk.zeros((2,)) + 1e-9, bk.zeros((2,))) is True
    assert bk.allclose(bk.zeros((2,)) + 1e-7, bk.zeros((2,))) is False


def test_linalg_norm_keepdims(any_backend: Backend):
    bk = any_backend
    norm = bk.linalg_norm(bk.ones((2, 2)), keepdims=True)
    assert tuple(norm.shape) == (1, 1)
    bk.assert_allclose(norm, bk.array([[2.0]]))


def test_array_discards_imaginary_part(real_backend: Backend):
    bk = real_backend
    complex_bk = bk.to_complex_backend()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x in [np.array([1 + 2j]), complex_bk.array([1 + 2j])]:
            bk.assert_allclose(bk.array(x), bk.array([1.0]))
