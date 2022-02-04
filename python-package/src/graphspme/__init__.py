import sys

import numpy
import scipy.sparse

from _graphspme import _prec_sparse, _cov_shrink_spd


if sys.version_info >= (3, 7):
    import numpy.typing

    NDArray = numpy.typing.NDArray
else:
    NDArray = numpy.ndarray


def prec_sparse(
    x: NDArray,
    Z: scipy.sparse.csr_matrix,
    cov_shrinkage: bool = True,
) -> scipy.sparse.csr_matrix:
    if not scipy.sparse.isspmatrix_csr(Z):
        raise ValueError(
            "Z matrix is not on csr (Compressed Sparse Row) format. "
            "This is usually solved by adding format='csr' to the"
            "matrix initialization, e.g. identity(n=10, format='csr')"
        )
    if len(x.shape) != 2:
        raise ValueError(
            f"x should be a two dimensional matrix, but has shape: {x.shape}"
        )
    _, n = x.shape
    if Z.shape != (n, n):
        raise ValueError(
            "If x has shape (p,n) then Z should have shape (n,n), "
            f"but got x.shape = {x.shape} and Z.shape = {Z.shape}"
        )
    return _prec_sparse(x, Z, cov_shrinkage)


def cov_shrink_spd(x: NDArray) -> NDArray:
    return _cov_shrink_spd(x)
