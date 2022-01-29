## License: GPL-3

# Stein-type covariance shrinkage estimate 
#

#' Asymptotic Stein-type shrinkage of SPD covariance estimate.
#'
#' \code{cov_shrink_spd} is the covariance estimator in \pkg{GraphSPME}.
#'
#' @param x an \eqn{nxp} matrix with \eqn{n-}observations of a \eqn{p-}vector
#'
#' @details
#' 
#' The standard frequentist covariance matrix estimate is calculated.
#' Using techniques of Touloumis (2015) the matrix is shrunk towards a 
#' sparse target matrix, set to be the diagonal matrix of the frequentist
#' estimate. The amount of shrinkage is found using asymptotic results
#' found in the reference paper.
#'
#' @return
#' Dense covariance matrix that is SPD.
#'
#' @seealso
#' \code{\link{prec_sparse}}
#'
#' @rdname cov_shrink_spd
#' @export
cov_shrink_spd <- function(x){
    # check input
    
    # calculate cov
    return(.cov_shrink_spd(x))
}
