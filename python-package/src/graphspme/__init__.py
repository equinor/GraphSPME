import sys

import numpy
import scipy.sparse

from _graphspme import (
    _prec_sparse,
    _cov_shrink_spd,
    _prec_nll,
    _prec_aic,
    _compute_amd_ordering,
    _cholesky_factor,
    _chol_to_precision,
    _dmrf,
    _dmrfL,
    _ddmrf,
    _ddmrfL,
    _ensure_eigenvalue_lower_bound,
)


if sys.version_info >= (3, 7):
    import numpy.typing

    NDArray = numpy.typing.NDArray
else:
    NDArray = numpy.ndarray


def prec_sparse(
    x: NDArray,
    Graph: scipy.sparse.csr_matrix,
    markov_order: int = 1,
    cov_shrinkage: bool = True,
    symmetrization: bool = True,
) -> scipy.sparse.csr_matrix:
    if not scipy.sparse.isspmatrix_csr(Graph):
        raise ValueError(
            "Graph matrix is not on csr (Compressed Sparse Row) format. "
            "This is usually solved by adding format='csr' to the"
            "matrix initialization, e.g. identity(n=10, format='csr')"
        )
    if len(x.shape) != 2:
        raise ValueError(
            f"x should be a two dimensional matrix, but has shape: {x.shape}"
        )
    _, n = x.shape
    if Graph.shape != (n, n):
        raise ValueError(
            "If x has shape (p,n) then Graph should have shape (n,n), "
            f"but got x.shape = {x.shape} and Graph.shape = {Graph.shape}"
        )
    if not isinstance(markov_order, int) or markov_order < 0:
        raise ValueError(
            "markov_order should be a non-negative integer, "
            f"but got markov_order = {markov_order}"
        )
    return _prec_sparse(x, Graph, markov_order, cov_shrinkage, symmetrization)


def cov_shrink_spd(x: NDArray) -> NDArray:
    return _cov_shrink_spd(x)


def prec_nll(x: NDArray, Prec: scipy.sparse.csr_matrix) -> float:
    return _prec_nll(x, Prec)


def prec_aic(x: NDArray, Prec: scipy.sparse.csr_matrix) -> float:
    return _prec_aic(x, Prec)


def compute_amd_ordering(A: scipy.sparse.csr_matrix) -> NDArray:
    return _compute_amd_ordering(A)


def cholesky_factor(
    A: scipy.sparse.csr_matrix, perm_indices: NDArray
) -> scipy.sparse.csr_matrix:
    return _cholesky_factor(A, perm_indices)


def chol_to_precision(
    L: scipy.sparse.csr_matrix, perm_indices: NDArray
) -> scipy.sparse.csr_matrix:
    return _chol_to_precision(L, perm_indices)


def dmrf(
    X: NDArray, Prec: scipy.sparse.csr_matrix, perm_indices: NDArray
) -> float:
    return _dmrf(X, Prec, perm_indices)


def dmrfL(
    X: NDArray, Prec: scipy.sparse.csr_matrix, perm_indices: NDArray
) -> float:
    return _dmrfL(X, Prec, perm_indices)


def ddmrf(
    X: NDArray,
    Prec: scipy.sparse.csr_matrix,
    perm_indices: NDArray,
    gradient_scale: float,
) -> NDArray:
    return _ddmrf(X, Prec, perm_indices, gradient_scale)


def ddmrfL(
    X: NDArray,
    Prec: scipy.sparse.csr_matrix,
    perm_indices: NDArray,
    gradient_scale: float,
) -> NDArray:
    return _ddmrfL(X, Prec, perm_indices, gradient_scale)


def ensure_eigenvalue_lower_bound(
    A: scipy.sparse.csc_matrix, eps: float = 1e-3, is_symmetric: bool = True
) -> scipy.sparse.csc_matrix:
    return _ensure_eigenvalue_lower_bound(A, eps, is_symmetric)
