## License: GPL-3

# Akaike Information Criterion for precision estimation
#

#' AIC for precision estimation.
#'
#' \code{prec_aic} gives the AIC-penalized Gaussian negative log-likelihood in \pkg{GraphSPME}.
#'
#' @param x an \eqn{nxp} matrix with \eqn{n-}observations of a \eqn{p-}vector
#' @param Prec a \eqn{pxp} sparse precision matrix.
#'
#' @details
#'
#' The AIC-penalized Gaussian negative log-likelihood given by
#' 
#' \deqn{AIC(\Lambda) = \frac{1}{2} tr(S\Lambda)-\log|\Lambda| + c )}{
#' AIC(\Lambda) = 0.5(tr(S\Lambda)-\log|\Lambda|) + c}
#'
#' @return
#' The value of the penalized negative log-likelihood.
#'
#' @seealso
#' \code{\link{prec_nll}}
#'
#' @rdname prec_aic
#' @export
prec_aic <- function(x, Prec) {
    return(.prec_aic(x, Prec))
}
