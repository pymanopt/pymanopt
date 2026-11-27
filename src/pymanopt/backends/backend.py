from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps
from typing import (
    Callable,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
import scipy.special

__all__ = ["TupleOrList", "DTypePrecision", "Backend", "DummyBackend"]

T = TypeVar("T")
TupleOrList = Union[list[T], tuple[T, ...]]


def not_implemented(function):
    @wraps(function)
    def inner(self, *args, **kwargs):
        raise NotImplementedError(
            f"Function '{function.__name__}' not implemented for backend {self}"
        )

    return inner


class DTypePrecision(Enum):
    """Enum for the dtype precision of the backend."""

    SINGLE = 32
    DOUBLE = 64


class Backend(ABC):
    """Abstract base class defining the interface for backends.

    A backend provides automatic differentiation of cost functions and the
    numerical functions used by manifolds.
    """

    ##########################################################################
    # Common attributes, properties and methods
    ##########################################################################
    _dtype: type

    @runtime_checkable  # to be able to check isinstance(x, bk.array_t)
    class array_t(Protocol):
        """Array type used for static type checks.

        It is supposed to represent either a numpy array, or a jax array, or a
        torch tensor, or a tensorflow tensor.
        """

        IndexType = Union[int, slice, None]

        def __getitem__(
            self, idx: Union[IndexType, tuple[IndexType, ...]]
        ) -> "Backend.array_t": ...

        def __len__(self) -> int: ...

        def __neg__(self) -> "Backend.array_t": ...

        def __add__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> "Backend.array_t": ...

        def __radd__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> "Backend.array_t": ...

        def __mul__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> "Backend.array_t": ...

        def __rmul__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> "Backend.array_t": ...

        def __matmul__(
            self, other: "Backend.array_t"
        ) -> "Backend.array_t": ...

        def __div__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> "Backend.array_t": ...

        def __truediv__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> "Backend.array_t": ...

        def __sub__(self, other: "Backend.array_t") -> "Backend.array_t": ...

        def __pow__(self, exponent: int) -> "Backend.array_t": ...

        @property
        def shape(self) -> tuple[int, ...]: ...

        def __eq__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> bool: ...

        def __leq__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> "Backend.array_t": ...

        def __geq__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> "Backend.array_t": ...

        def __lt__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> "Backend.array_t": ...

        def __gt__(
            self, other: "Union[int, float, Backend.array_t]"
        ) -> "Backend.array_t": ...

        @property
        def T(self) -> "Backend.array_t": ...

    @property
    @abstractmethod
    def dtype(self) -> type: ...

    @property
    @abstractmethod
    def dtype_precision(self) -> DTypePrecision: ...

    @property
    @abstractmethod
    def is_dtype_real(self) -> bool: ...

    @staticmethod
    @abstractmethod
    def DEFAULT_REAL_DTYPE() -> type: ...

    @staticmethod
    @abstractmethod
    def DEFAULT_COMPLEX_DTYPE() -> type: ...

    @abstractmethod
    def __repr__(self) -> str: ...

    def __eq__(self, other):
        return repr(self) == repr(other)

    @abstractmethod
    def to_complex_backend(self) -> "Backend": ...

    @abstractmethod
    def to_real_backend(self) -> "Backend": ...

    ##############################################################################
    # Autodiff methods
    ##############################################################################

    @abstractmethod
    def generate_gradient_operator(self, function, num_arguments) -> Callable:
        """Creates a function to compute gradients of a function.

        Args:
            function: A callable.
            num_arguments: The number of arguments that ``function`` expects.

        Returns:
            A Python callable of the gradient of `function` accepting arguments
            according to the signature defined by `arguments`.
        """

    @abstractmethod
    def generate_hessian_operator(self, function, num_arguments) -> Callable:
        """Creates a function to compute Hessian-vector products of a function.

        Args:
            function: A callable.
            num_arguments: The number of arguments that ``function`` expects.

        Returns:
            A Python callable evaluating the Hessian-vector product of
            ``function`` accepting arguments according to the signature defined
            by ``arguments``.
            The returned callable accepts a point of evaluation as a sequence
            of length ``num_arguments``, as well as a vector of the same shape
            that is right-multiplied to the Hessian.
        """

    ##############################################################################
    # Numerics methods
    ##############################################################################

    @not_implemented
    def abs(self, array: array_t) -> array_t:
        """Element-wise absolute value, like :func:`numpy.absolute`."""

    @not_implemented
    def all(self, array: array_t) -> bool:
        """Test whether all elements are true, like :func:`numpy.all`."""

    @not_implemented
    def allclose(
        self,
        array_a: array_t,
        array_b: array_t,
        rtol: float,
        atol: float,
    ) -> bool:
        """Test closeness with rtol and atol, like :func:`numpy.allclose`."""

    @not_implemented
    def any(self, array: array_t) -> Optional[bool]:
        """Test whether any element is true, like :func:`numpy.any`."""

    @not_implemented
    def arange(
        self,
        start: int,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> array_t:
        """Evenly spaced values excluding stop, like :func:`numpy.arange`."""

    @not_implemented
    def arccos(self, array: array_t) -> array_t:
        """Element-wise arc cosine, like :func:`numpy.arccos`."""

    @not_implemented
    def arccosh(self, array: array_t) -> array_t:
        """Inverse hyperbolic cosine, like :func:`numpy.arccosh`."""

    @not_implemented
    def arctan(self, array: array_t) -> array_t:
        """Element-wise arc tangent, like :func:`numpy.arctan`."""

    @not_implemented
    def arctanh(self, array: array_t) -> array_t:
        """Inverse hyperbolic tangent, like :func:`numpy.arctanh`."""

    @not_implemented
    def argmin(self, array: array_t) -> array_t:
        """Flat index of the minimum, like :func:`numpy.argmin`."""

    @not_implemented
    def argsort(self, array: array_t) -> array_t:
        """Indices that sort the last axis, like :func:`numpy.argsort`."""

    @not_implemented
    def array(self, array: array_t) -> array_t:
        """Convert to the backend's dtype, like :func:`numpy.asarray`."""

    @not_implemented
    def assert_allclose(
        self,
        array_a: array_t,
        array_b: array_t,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        """Raise ``ValueError`` unless arrays are close within tolerances."""

    @not_implemented
    def assert_equal(self, array_a: array_t, array_b: array_t) -> None:
        """Raise ``ValueError`` unless arrays are exactly equal."""

    @not_implemented
    def concatenate(
        self,
        arrays: TupleOrList[array_t],
        axis: int = 0,
    ) -> array_t:
        """Join arrays on an existing axis, like :func:`numpy.concatenate`."""

    @not_implemented
    def conjugate(self, array: array_t) -> array_t:
        """Element-wise complex conjugate, like :func:`numpy.conjugate`."""

    def conjugate_transpose(self, array: array_t) -> array_t:
        """Conjugate and swap the last two axes, preserving leading axes."""
        return self.conjugate(self.transpose(array))

    @not_implemented
    def cos(self, array: array_t) -> array_t:
        """Element-wise cosine, like :func:`numpy.cos`."""

    @not_implemented
    def diag(self, array: array_t) -> array_t:
        """Extract or construct a diagonal, like :func:`numpy.diag`."""

    @not_implemented
    def diagonal(
        self,
        array: array_t,
        axis1: int,
        axis2: int,
    ) -> array_t:
        """Extract the diagonal along two axes, like :func:`numpy.diagonal`."""

    @not_implemented
    def eps(self) -> float:
        """Gap from 1 to the next larger value at the backend's precision."""

    @not_implemented
    def exp(self, array: array_t) -> array_t:
        """Element-wise exponential, like :func:`numpy.exp`."""

    @not_implemented
    def expand_dims(self, array: array_t, axis: int) -> array_t:
        """Insert a length-one axis, like :func:`numpy.expand_dims`."""

    @not_implemented
    def eye(self, size: int) -> array_t:
        """Identity in the backend's dtype, like :func:`numpy.eye`."""

    @not_implemented
    def hstack(self, arrays: TupleOrList[array_t]) -> array_t:
        """Stack arrays horizontally, like :func:`numpy.hstack`."""

    def herm(self, array: array_t) -> array_t:
        """Hermitian part ``(A + A*) / 2`` over the last two axes."""
        return 0.5 * (array + self.conjugate_transpose(array))

    @not_implemented
    def imag(self, array: array_t) -> array_t:
        """Element-wise imaginary part, like :func:`numpy.imag`."""

    @not_implemented
    def iscomplexobj(self, array: array_t) -> bool:
        """Test for a complex dtype, like :func:`numpy.iscomplexobj`."""

    @not_implemented
    def isnan(self, array: array_t) -> array_t:
        """Element-wise test for NaN, like :func:`numpy.isnan`."""

    @not_implemented
    def isrealobj(self, array: array_t) -> bool:
        """Test for a non-complex dtype, like :func:`numpy.isrealobj`."""

    @not_implemented
    def linalg_cholesky(self, array: array_t) -> array_t:
        """Lower Cholesky factors, like :func:`numpy.linalg.cholesky`."""

    @not_implemented
    def linalg_det(self, array: array_t) -> array_t:
        """Determinant on the last two axes, like :func:`numpy.linalg.det`."""

    @not_implemented
    def linalg_eigh(self, array: array_t) -> tuple[array_t, array_t]:
        """Hermitian eigenpairs, like :func:`numpy.linalg.eigh`."""

    @not_implemented
    def linalg_eigvalsh(
        self,
        array_x: array_t,
        array_y: Optional[array_t] = None,
    ) -> array_t:
        """Hermitian eigenvalues over the last two axes.

        Args:
            array_x: Symmetric or Hermitian matrices.
            array_y: Positive definite matrices for the generalized problem
                ``array_x @ v = w * array_y @ v``; ``None`` uses the identity.
        """

    @not_implemented
    def linalg_expm(
        self,
        array: array_t,
        symmetric: bool = False,
    ) -> array_t:
        """Matrix exponential over the last two axes, preserving leading axes.

        Args:
            array: Square matrices.
            symmetric: Assume symmetric or Hermitian matrices and use their
                eigendecomposition.
        """

    @not_implemented
    def linalg_inv(self, array: array_t) -> array_t:
        """Inverse over the last two axes, like :func:`numpy.linalg.inv`."""

    @not_implemented
    def linalg_matrix_rank(self, array: array_t) -> int:
        """Numerical matrix rank, like :func:`numpy.linalg.matrix_rank`."""

    @not_implemented
    def linalg_logm(
        self,
        array: array_t,
        positive_definite: bool = False,
    ) -> array_t:
        """Matrix logarithm over the last two axes, preserving leading axes.

        Args:
            array: Square matrices.
            positive_definite: Assume symmetric or Hermitian positive definite
                matrices and use their eigendecomposition.
        """

    @not_implemented
    def linalg_norm(
        self,
        array: array_t,
        ord: Union[  # noqa: A002 pylint: disable=redefined-builtin
            int, Literal["fro"], None
        ] = None,  # noqa: A002 pylint: disable=redefined-builtin
        axis: Union[int, TupleOrList[int], None] = None,
        keepdims: bool = False,
    ) -> array_t:
        """Vector or matrix norm, like :func:`numpy.linalg.norm`."""

    @not_implemented
    def linalg_qr(self, array: array_t) -> tuple[array_t, array_t]:
        """Reduced QR on the last two axes, with nonnegative diagonal of R."""

    @not_implemented
    def linalg_solve(
        self,
        array_a: array_t,
        array_b: array_t,
    ) -> array_t:
        """Solve ``array_a @ x = array_b``, like :func:`numpy.linalg.solve`."""

    @not_implemented
    def linalg_solve_continuous_lyapunov(
        self,
        array_a: array_t,
        array_q: array_t,
    ) -> array_t:
        """Solve ``A @ X + X @ A* = Q``, where ``A*`` is the adjoint of A."""

    @not_implemented
    def linalg_svd(
        self,
        array: array_t,
        full_matrices: bool = True,
    ) -> tuple[array_t, array_t, array_t]:
        """Return ``U, s, Vh``, like :func:`numpy.linalg.svd`."""

    @not_implemented
    def linalg_svdvals(self, array: array_t) -> array_t:
        """Singular values, like :func:`numpy.linalg.svdvals`."""

    @not_implemented
    def log(self, array: array_t) -> array_t:
        """Element-wise natural logarithm, like :func:`numpy.log`."""

    @not_implemented
    def log10(self, array: array_t) -> array_t:
        """Element-wise base-10 logarithm, like :func:`numpy.log10`."""

    @not_implemented
    def logical_not(self, array: array_t) -> array_t:
        """Element-wise logical negation, like :func:`numpy.logical_not`."""

    @not_implemented
    def logspace(self, start: float, stop: float, num: int) -> array_t:
        """Base-10 spacing in backend dtype, like :func:`numpy.logspace`."""

    def matvec(self, A: array_t, x: array_t) -> array_t:
        """Multiply a matrix by a vector using ``A @ x``."""
        return A @ x

    def matmul(self, A: array_t, B: array_t) -> array_t:
        """Matrix product, broadcasting batches, like :func:`numpy.matmul`."""
        return A @ B

    def multieye(self, k: int, n: int) -> array_t:
        """Stack k identity matrices into shape ``(k, n, n)``."""
        return self.tile(self.eye(n), (k, 1, 1))

    @not_implemented
    def ndim(self, array: array_t) -> int:
        """Number of array dimensions, like :func:`numpy.ndim`."""

    newaxis = None

    @not_implemented
    def ones(self, shape: TupleOrList[int]) -> array_t:
        """Ones in the backend's dtype, like :func:`numpy.ones`."""

    @not_implemented
    def ones_bool(self, shape: TupleOrList[int]) -> array_t:
        """Boolean array filled with ``True``."""

    pi = np.pi

    @not_implemented
    def prod(self, array: array_t) -> float:
        """Product of all elements, like :func:`numpy.prod`."""

    @not_implemented
    def random_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Union[int, TupleOrList[int], None] = None,
    ) -> array_t:
        """Normal samples in the backend's dtype.

        Complex samples have independent real and imaginary parts, each with
        the given mean and standard deviation.

        Args:
            loc: Mean of each real component.
            scale: Standard deviation of each real component.
            size: Sample shape; ``None`` draws a single sample.
        """

    def random_randn(self, *dims: int) -> array_t:
        """Standard normal samples of shape dims, via :meth:`random_normal`."""
        return self.random_normal(loc=0.0, scale=1.0, size=dims)

    @not_implemented
    def random_uniform(
        self, size: Union[int, TupleOrList[int], None] = None
    ) -> array_t:
        """Uniform samples in ``[0, 1)`` in the backend's dtype.

        Complex samples have independent parts in ``[0, 1)``.

        Args:
            size: Sample shape; ``None`` draws a single sample.
        """

    @not_implemented
    def real(self, array: array_t) -> array_t:
        """Element-wise real part, like :func:`numpy.real`."""

    @not_implemented
    def reshape(
        self,
        array: array_t,
        shape: TupleOrList[int],
    ) -> array_t:
        """Reshape, preserving element order, like :func:`numpy.reshape`."""

    # TODO: seterr
    # def seterr(all=None):
    #     np.seterr(all=all)
    #     if all == 'raise':
    #         try:
    #             import torch
    #             torch.autograd.set_detect_anomaly(True)
    #         except ImportError:
    #             ...
    #         try:
    #             import jax
    #             jax.config.update("jax_debug_nans", True)
    #         except ImportError:
    #             ...
    #         try:
    #             import tensorflow as tf
    #             tf.debugging.enable_check_numerics()
    #         except ImportError:
    #             ...

    @not_implemented
    def sin(self, array: array_t) -> array_t:
        """Element-wise sine, like :func:`numpy.sin`."""

    @not_implemented
    def sinc(self, array: array_t) -> array_t:
        """Normalized sinc, ``sin(pi*x)/(pi*x)``, like :func:`numpy.sinc`."""

    def skew(self, array: array_t) -> array_t:
        """Skew-symmetric part ``(A - A.T) / 2`` over the last two axes."""
        return 0.5 * (array - self.transpose(array))

    def skewh(self, array: array_t) -> array_t:
        """Skew-Hermitian part ``(A - A*) / 2`` over the last two axes."""
        return 0.5 * (array - self.conjugate_transpose(array))

    @not_implemented
    def sort(
        self,
        array: array_t,
        descending: bool = False,
    ) -> array_t:
        """Sort the last axis in ascending or descending order."""

    def special_comb(self, n: int, k: int):
        """Ways to choose k from n, like :func:`scipy.special.comb`."""
        return scipy.special.comb(n, k)

    @not_implemented
    def sqrt(self, array: array_t) -> array_t:
        """Element-wise square root, like :func:`numpy.sqrt`."""

    @not_implemented
    def squeeze(self, array: array_t) -> array_t:
        """Remove all length-one axes, like :func:`numpy.squeeze`."""

    @not_implemented
    def stack(
        self,
        arrays: TupleOrList[array_t],
        axis: int = 0,
    ) -> array_t:
        """Join arrays along a new axis, like :func:`numpy.stack`."""

    @not_implemented
    def sum(
        self,
        array: array_t,
        axis: Union[int, TupleOrList[int], None] = None,
        keepdims: bool = False,
    ) -> array_t:
        """Sum over axes, or all axes if None, like :func:`numpy.sum`."""

    def sym(self, array: array_t) -> array_t:
        """Symmetric part ``(A + A.T) / 2`` over the last two axes."""
        return 0.5 * (array + self.transpose(array))

    @not_implemented
    def tan(self, array: array_t) -> array_t:
        """Element-wise tangent, like :func:`numpy.tan`."""

    @not_implemented
    def tanh(self, array: array_t) -> array_t:
        """Element-wise hyperbolic tangent, like :func:`numpy.tanh`."""

    @not_implemented
    def tensordot(
        self,
        a: array_t,
        b: array_t,
        axes: int = 2,
    ) -> array_t:
        """Tensor contraction, like :func:`numpy.tensordot`.

        Args:
            a: First input array.
            b: Second input array.
            axes: Number of trailing axes of a to contract with leading axes
                of b.
        """

    @not_implemented
    def tile(
        self,
        array: array_t,
        reps: Union[int, TupleOrList[int]],
    ) -> array_t:
        """Repeat an array along each axis, like :func:`numpy.tile`."""

    @not_implemented
    def trace(self, array: array_t) -> array_t:
        """Sum the diagonal on the last two axes, preserving leading axes."""

    @not_implemented
    def transpose(self, array: array_t) -> array_t:
        """Swap the last two axes without conjugation; keep leading axes."""

    @not_implemented
    def triu(self, array: array_t, k: int = 0) -> array_t:  # type: ignore
        """Upper triangle on the last two axes, like :func:`numpy.triu`.

        Args:
            array: Input matrices.
            k: Diagonal offset; positive is above the main diagonal.
        """

    @not_implemented
    def vstack(self, arrays: TupleOrList[array_t]) -> array_t:
        """Stack arrays vertically, like :func:`numpy.vstack`."""

    @not_implemented
    def where(
        self,
        condition: array_t,
        x: Optional[array_t],
        y: Optional[array_t],
    ) -> Union[array_t, tuple[array_t, ...]]:
        """Select elements or find true indices, like :func:`numpy.where`.

        Args:
            condition: Boolean selection mask.
            x: Values at true entries; use ``None`` with ``y=None`` to return
                indices of true entries instead.
            y: Values at false entries; must be supplied together with x.
        """

    @not_implemented
    def zeros(self, shape: TupleOrList[int]) -> array_t:
        """Zeros in the backend's dtype, like :func:`numpy.zeros`."""

    @not_implemented
    def zeros_bool(self, shape: TupleOrList[int]) -> array_t:
        """Boolean array filled with ``False``."""

    @not_implemented
    def zeros_like(self, array: array_t) -> array_t:
        """Zeros in backend dtype, like :func:`numpy.zeros_like`."""


class DummyBackend(Backend):
    """Placeholder backend of manifolds created without a backend.

    This class should not be used directly.
    """

    @property
    def dtype(self): ...

    @property
    def dtype_precision(self): ...

    @property
    def is_dtype_real(self): ...

    @staticmethod
    def DEFAULT_REAL_DTYPE(): ...

    @staticmethod
    def DEFAULT_COMPLEX_DTYPE(): ...

    def __repr__(self):
        return "DummyBackend"

    def __eq__(self, other):
        return repr(self) == repr(other)

    def to_complex_backend(self) -> "DummyBackend":
        return self

    def to_real_backend(self) -> "DummyBackend":
        return self

    def generate_gradient_operator(self, function, num_arguments):
        raise NotImplementedError(
            "DummyBackend provides no gradient operator generation"
        )

    def generate_hessian_operator(self, function, num_arguments):
        raise NotImplementedError(
            "DummyBackend provides no Hessian operator generation"
        )
