import sys

import numpy
import scipy.sparse

from _graphspme import _prec_sparse, _cov_shrink_spd, _prec_nll, _prec_aic
from _graphspme import _ldl_fbmod


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
    cholmod_adjustment: float = 1e-3,
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
    prec = _prec_sparse(x, Graph, markov_order, cov_shrinkage, symmetrization)
    if cholmod_adjustment:
        prec = _ldl_fbmod(prec, 1e-3, symmetrization)
    return prec


def cov_shrink_spd(x: NDArray) -> NDArray:
    return _cov_shrink_spd(x)


def prec_nll(x: NDArray, Prec: scipy.sparse.csr_matrix) -> float:
    return _prec_nll(x, Prec)


def prec_aic(x: NDArray, Prec: scipy.sparse.csr_matrix) -> float:
    return _prec_aic(x, Prec)
