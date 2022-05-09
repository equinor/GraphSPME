## License: GPL-3

# Gaussian negative log-likelihood relevant for precision estimation
#

#' Gaussian negative log-likelihood relevant for precision estimation.
#'
#' \code{prec_nll} gives the Gaussian negative log-likelihood in \pkg{GraphSPME}.
#'
#' @param x an \eqn{nxp} matrix with \eqn{n-}observations of a \eqn{p-}vector
#' @param Prec a \eqn{pxp} sparse precision matrix.
#'
#' @details
#'
#' The Gaussian negative log-likelihood given by
#' 
#' \deqn{l(\Lambda) = \frac{1}{2} tr(S\Lambda)-\log|\Lambda| )}{
#' l(\Lambda) = 0.5(tr(S\Lambda)-\log|\Lambda|)}
#'
#' @return
#' The value of the negative log-likelihood.
#'
#' @seealso
#' \code{\link{prec_aic}}
#'
#' @rdname prec_nll
#' @export
prec_nll <- function(x, Prec) {
    return(.prec_nll(x, Prec))
}
