// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// _cov_shrink_spd
Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> _cov_shrink_spd(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& x);
RcppExport SEXP _GraphSPME__cov_shrink_spd(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(_cov_shrink_spd(x));
    return rcpp_result_gen;
END_RCPP
}
// _sparse_matrix_inverse
Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> _sparse_matrix_inverse(Eigen::SparseMatrix<double>& A);
RcppExport SEXP _GraphSPME__sparse_matrix_inverse(SEXP ASEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::SparseMatrix<double>& >::type A(ASEXP);
    rcpp_result_gen = Rcpp::wrap(_sparse_matrix_inverse(A));
    return rcpp_result_gen;
END_RCPP
}
// _prec_sparse
Eigen::SparseMatrix<double> _prec_sparse(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& x, Eigen::SparseMatrix<double>& Z, bool cov_shrinkage);
RcppExport SEXP _GraphSPME__prec_sparse(SEXP xSEXP, SEXP ZSEXP, SEXP cov_shrinkageSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& >::type x(xSEXP);
    Rcpp::traits::input_parameter< Eigen::SparseMatrix<double>& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< bool >::type cov_shrinkage(cov_shrinkageSEXP);
    rcpp_result_gen = Rcpp::wrap(_prec_sparse(x, Z, cov_shrinkage));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_GraphSPME__cov_shrink_spd", (DL_FUNC) &_GraphSPME__cov_shrink_spd, 1},
    {"_GraphSPME__sparse_matrix_inverse", (DL_FUNC) &_GraphSPME__sparse_matrix_inverse, 1},
    {"_GraphSPME__prec_sparse", (DL_FUNC) &_GraphSPME__prec_sparse, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_GraphSPME(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}