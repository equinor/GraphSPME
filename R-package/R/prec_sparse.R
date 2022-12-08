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
#' ensured through symmetry conversion. See details for more.
#' @param cholmod_adjustment at the end of estimation performs the modified 
#' Cholesky decomposition of Cheng & Higham (1998), using \code{cholmod_adjustment}
#' as \eqn{\epsilon} as treshold. See details for more.
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
#' is returned. Note that symmetrization is always performed before modified Cholesky.
#' 
#' If \code{cholmod_adjustment} is numeric with some \eqn{\epsilon} then an 
#' adjustment of the precision
#' with block-LDL decomposition
#' \eqn{\Pi\hat{\Lambda}\Pi^\top=LDL^\top}
#' (Ashcraft et. al, 1998)
#' is performed as
#' \deqn{\hat{\Lambda}=\Pi^\top LD_+L^\top \Pi}
#' where the block-diagonal D_+ is modified to have 
#' eigenvalues \eqn{max(\epsilon, \lambda_{i})}.
#' All computations happen in a sparsity aware manner.
#' Note that the eigenvalue adjustment of \eqn{D} approximates the result of 
#' Higham (1988) calculating a nearest SP(S)D matrix in Frobenius norm as 
#' \eqn{A_{spsd}=0.5(H+U)} of some matrix \eqn{A} where
#' \eqn{H=0.5(A+A^\top)} is the nearest symmetric matrix and
#' \eqn{U} is the polar factor in the polar decomposition 
#' \eqn{H=PU}.
#' These computations are iteratively employed on the blocks of \eqn{D},
#' but not directly on the full matrix \eqn{\hat{\Lambda}} due to dense
#' computations from working with the spectral decomposition.
#' If the problem is small, the computations may be accessed through
#' the function \code{.ensure_eigenvalue_lower_bound()}.
#' 
#' The algorithm should be robust in the face of extremely high dimensions. 
#' 
#' @return
#' Sparse precision matrix that is SPD.
#'
#' @seealso
#' \code{\link{cov_shrink_spd}}
#' 
#' @references
#' 
#' Berent Ånund Strømnes Lunde, Feda Curic and Sondre Sortland,
#' "GraphSPME: Markov Precision Matrix Estimation and Asymptotic Stein-Type Shrinkage", 2022, 
#' \url{https://arxiv.org/abs/2205.07584}
#'
#' Nicholas J. Higham,
#' "Computing a nearest symmetric positive semidefinite matrix", 1988, 
#' \url{https://www.sciencedirect.com/science/article/pii/0024379588902236}
#' 
#' Sheung Hun Cheng & Nicholas J. Higham
#' "A modified Cholesky algorithm based on a symmetric indefinite factorization", 1998
#' \url{https://epubs.siam.org/doi/10.1137/S0895479896302898}
#' 
#' C. Ashcraft, R. G. Grimes, and J. G. Lewis
#' "Accurate symmetric indefinite linear equation solvers", 1998
#' \url{https://epubs.siam.org/doi/10.1137/S0895479896296921}
#'
#' @rdname prec_sparse
#' @export
prec_sparse <- function(
        X, 
        Graph, 
        markov_order = 1, 
        shrinkage = TRUE, 
        symmetrization = TRUE, 
        cholmod_adjustment = 1e-3
        ) {
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
  prec = .prec_sparse(X, Graph, markov_order, shrinkage, symmetrization)
  
  # Eigenvalue adjustments
  if(is.numeric(cholmod_adjustment)){
      is_symmetric = symmetrization
      prec <- .ldl_fbmod(prec, cholmod_adjustment, is_symmetric)
  }
  
  return(prec)
}
