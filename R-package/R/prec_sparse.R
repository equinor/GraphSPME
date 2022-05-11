## License: GPL-3

# Graphical Sparse Precision Matrix Estimation
#

#' Graphical Sparse Precision Matrix Estimation
#'
#' \code{prec_sparse} is the precision matrix estimator in \pkg{GraphSPME}.
#'
#' @param X an \eqn{nxp} matrix with \eqn{n-}observations of a \eqn{p-}vector
#' @param Graph a \eqn{pxp} sparse matrix representing a graph with vertices
#' corresponding to columns of \code{X} and edges to direct connections
#' (neighborhood) between columns of \code{X}
#' @param markov_order non-negative int representing the markov order of data
#' \code{X} wrt \code{Graph}
#' @param shrinkage if shrinkage should be applied to
#' block-frequentist-covariance see details for more
#' @param symmetrization if symmetry of the precision estimate should be
#' ensured through symmetry conversion. See details for more
#'
#' @details
#'
#' The algorithm utilizes the knowledge of the graph provided through
#' \code{Graph} and \code{markov_order} to create sparse matrix \code{Z} indicating the
#' non-zero elements of the precision matrix of \code{X}.
#'
#' The resulting sparse matrix \code{Z} is then utilized to estimate a sparse
#' precision matrix using the algorithm of Le (2021).
#'
#' If \code{shrinkage=TRUE} then
#' the method of Le (2021) is modified by the method in
#' Lunde (2022) to iteratively employ asymptotic shrinkage when building
#' the sparse precision matrix estimator.
#'
#' Employing shrinkage ensures numerical stability while improving the resulting
#' precision/covariance estimate. This is particularly noticeable when the
#' problem involves many neighbors for each vertex and/or has a high Markov
#' order.
#'
#' If \code{symmetrization=TRUE} then the symmetry conversion
#' \deqn{\hat{\Lambda}=0.5(\tilde{\Lambda}+\tilde{\Lambda}^\top)}{
#' 0.5(\Lambda+\Lambda^T)}
#' is returned.
#'
#' The algorithm should be robust in the face of extremely high dimensions.
#'
#' @return
#' Sparse precision matrix that is SPD.
#'
#' @seealso
#' \code{\link{cov_shrink_spd}}
#'
#' @rdname prec_sparse
#' @export
prec_sparse <- function(X, Graph, markov_order = 1, shrinkage = TRUE, symmetrization = TRUE) {
  # check xtype
  if (!is.matrix(X)) {
    stop("X must be a matrix")
  }
  # dimensions
  if (!all(ncol(X) == dim(Graph))) {
    stop("ncol(X) must equal both dim(Graph)")
  }
  # sparsity type
  if (!(class(Graph) == "dgCMatrix")) {
    stop("Graph must be a sparse matrix of type dgCMatrix")
  }
  # markov order
  is.wholenumber <-
    function(x, tol = .Machine$double.eps^0.5) abs(x - round(x)) < tol
  if (!is.numeric(markov_order) || !is.wholenumber(markov_order) || markov_order < 0) {
    stop("markov_order must be a non-negative whole number")
  }

  # calculate precision
  return(.prec_sparse(X, Graph, markov_order, shrinkage, symmetrization))
}
