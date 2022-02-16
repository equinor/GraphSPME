## License: GPL-3

# Graphical Sparse Precision Matrix Estimation
#

#' Graphical Sparse Precision Matrix Estimation
#'
#' \code{prec_sparse} is the precision matrix estimator in \pkg{GraphSPME}.
#'
#' @param x an \eqn{nxp} matrix with \eqn{n-}observations of a \eqn{p-}vector
#' @param Z a sparse matrix indicating non-zero elements of the precision matrix
#' @param shrinkage if shrinkage should be applied to block-frequentist-covariance see details for more
#'
#' @details
#' 
#' The algorithm utilizes the knowledge of the sparsity provided through
#' \code{Z}, using the algorithm of Le (2021). If \code{shrinkage=TRUE} then
#' the method of Le (2021) is modified by the method in 
#' Lunde (2022) to iteratively employ asymptotic shrinkage when building
#' the sparse precision matrix estimator.
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
prec_sparse <- function(x, Z, shrinkage=TRUE){
    # check xtype
    if(!is.matrix(x)){
        stop("x must be a matrix")   
    }
    # dimensions
    if(!all(ncol(x)==dim(Z))){
        stop("ncol(x) must equal both dim(Z)")
    }
    #sparsity type
    if(!(class(Z) == "dgCMatrix")){
        stop("Z must be a sparse matrix of type dgCMatrix")
    }

    # calculate precision
    return(.prec_sparse(x, Z, shrinkage))
}
