## License: GPL-3

#' \code{GraphSPME} is a package for sparse precision matrix estimation through
#' a graph.
#' It employs the method of Lunde 2022, imported through the \code{graph_spme} function.
#' 
#' Important functions:
#' 
#' \itemize{
#' \item \code{\link{prec_sparse}}: Sparse precision matrix estimation through a graph
#' \item \code{\link{cov_shrink_spd}}: Asymptotic Stein-type shrinkage of SPD covariance estimate
#' }
#' 
#' See individual function documentation for usage.
#'
#' @docType package
#' @name GraphSPME
#' @title Graphical Sparse Precision Matrix Estimation
#'
#' @author Berent Ånund Strømnes Lunde
#' @useDynLib GraphSPME, .registration = TRUE
#' @import Rcpp RcppEigen
#' @importFrom Rcpp evalCpp
NULL