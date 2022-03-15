import pytest
import numpy as np
from scipy import sparse

import graphspme


def test_prec_sparse_non_csr():
    x = np.tile(np.ones([4]), (4, 1))
    Graph = sparse.diags([1, 1, 1, 1], 0, format="csc")
    with pytest.raises(
        ValueError,
        match=r".*Graph matrix is not on csr \(Compressed Sparse Row\) format.*",
    ):
        graphspme.prec_sparse(x, Graph)


@pytest.mark.parametrize(
    "x", [np.ones(shape=(5,)), np.random.normal(0, 1, size=(5, 3, 5))]
)
def test_prec_sparse_x_wrong_dimensions(x):
    Graph = sparse.diags([1, 1, 1], 0, format="csr")
    with pytest.raises(ValueError, match=r".*x should be a two dimensional matrix.*"):
        graphspme.prec_sparse(x, Graph)


def test_prec_sparse_dimension_mismatch():
    x = np.tile(np.ones([4]), (4, 1))
    Graph = sparse.diags([1, 1, 1], 0, format="csr")
    with pytest.raises(
        ValueError, match=r".*but got x.shape = \(4, 4\) and Graph.shape = \(3, 3\)"
    ):
        graphspme.prec_sparse(x, Graph)
