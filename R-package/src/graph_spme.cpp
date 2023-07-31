// Main functions for GraphSPME
// License: GPL-3

#include <Rcpp.h>
#include "graph_spme.hpp"

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins("cpp11")]]

/*
 * Covariance shrinkage estimate as specified in Touloumis (2015)
 */
// [[Rcpp::export(.cov_shrink_spd)]]
Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> _cov_shrink_spd(
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& x
){
    return cov_shrink_spd(x);
}

/*
 *  Sparse precision matrix inverse
 *  Employs sparse cholesky
 */
// [[Rcpp::export(.sparse_matrix_inverse)]]
Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> _sparse_matrix_inverse(Eigen::SparseMatrix<double>& A){
    return sparse_matrix_inverse(A);
}

/*
 * Graphical sparse precision matrix estimation
 * as defined in Le (2021)
 * and possibilities of using covariance shrinkage from Lunde etal (2022?)
 */
// [[Rcpp::export(.prec_sparse)]]
Eigen::SparseMatrix<double> _prec_sparse(
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& x,
        Eigen::SparseMatrix<double>& Neighbours,
        int markov_order=1,
        bool cov_shrinkage=true,
        bool symmetrization=true
){
    return prec_sparse(x, Neighbours, markov_order, cov_shrinkage, symmetrization);
}

// [[Rcpp::export(.cov_ml)]]
Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> _cov_ml(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& x){
    return cov_ml(x);
}

// [[Rcpp::export(.create_bi)]]
Eigen::SparseMatrix<double> _create_bi(Eigen::SparseMatrix<double>& Z, int j) {
    return create_bi(Z, j);
}

// [[Rcpp::export(.get_precision_nonzero)]]
Eigen::SparseMatrix<double> _get_precision_nonzero(Eigen::SparseMatrix<double>& Graph, int markov_order) {
    return get_precision_nonzero(Graph, markov_order);
}

// [[Rcpp::export(.prec_nll)]]
double _prec_nll(
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& X,
        Eigen::SparseMatrix<double>& Prec
        ) {
    return prec_nll(X, Prec);
}

// [[Rcpp::export(.prec_aic)]]
double _prec_aic(
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& X,
        Eigen::SparseMatrix<double>& Prec
) {
    return prec_aic(X, Prec);
}

// [[Rcpp::export(.compute_amd_ordering)]]
Eigen::Matrix<int,Eigen::Dynamic,1> _compute_amd_ordering(
        Eigen::SparseMatrix<double> &A
) {
    return compute_amd_ordering(A);
}

// [[Rcpp::export(.cholesky_factor)]]
Eigen::SparseMatrix<double> __cholesky_factor(
        Eigen::SparseMatrix<double> &P,
        Eigen::Matrix<int,Eigen::Dynamic,1> perm_indices
) {
    return cholesky_factor(P, perm_indices);
}

// [[Rcpp::export(.chol_to_precision)]]
Eigen::SparseMatrix<double> _chol_to_precision(
        Eigen::SparseMatrix<double> &L,
        Eigen::Matrix<int,Eigen::Dynamic,1> perm_indices
) {
    return chol_to_precision(L, perm_indices);
}

// [[Rcpp::export(.dmrf)]]
double __dmrf(
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& X,
        Eigen::SparseMatrix<double>& Prec,
        Eigen::Matrix<int,Eigen::Dynamic,1> perm_indices
) {
    return dmrf(X, Prec, perm_indices);
}

// [[Rcpp::export(.dmrfL)]]
double __dmrfL(
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& X,
        Eigen::SparseMatrix<double>& L,
        Eigen::Matrix<int,Eigen::Dynamic,1> perm_indices
) {
    return dmrfL(X, L, perm_indices);
}

// [[Rcpp::export(.dmrf_grad)]]
Eigen::Matrix<double,Eigen::Dynamic,1> _dmrf_grad(
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& X,
        Eigen::SparseMatrix<double>& Prec,
        Eigen::SparseMatrix<double>& grad_elements_pick
) {
    return dmrf_grad(X, Prec, grad_elements_pick);
}

// [[Rcpp::export(.dmrf_hess)]]
Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> _dmrf_hess(
        Eigen::SparseMatrix<double>& Prec,
        Eigen::SparseMatrix<double>& grad_elements_pick
) {
    return dmrf_hess(Prec, grad_elements_pick);
}

// [[Rcpp::export(.dmrfL_grad)]]
Eigen::Matrix<double,Eigen::Dynamic,1> _dmrfL_grad(
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& X,
        Eigen::SparseMatrix<double>& L,
        Eigen::SparseMatrix<double>& grad_elements_pick,
        Eigen::Matrix<int,Eigen::Dynamic,1> perm_indices
) {
    return dmrfL_grad(X, L, grad_elements_pick, perm_indices);
}

// [[Rcpp::export(.dmrfL_hess)]]
Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> _dmrfL_hess(
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& X,
        Eigen::SparseMatrix<double>& L,
        Eigen::SparseMatrix<double>& grad_elements_pick,
        Eigen::Matrix<int,Eigen::Dynamic,1> perm_indices
) {
    return dmrfL_hess(X, L, grad_elements_pick, perm_indices);
}

// [[Rcpp::export(.ensure_eigenvalue_lower_bound)]]
Eigen::SparseMatrix<double> _ensure_eigenvalue_lower_bound(
        Eigen::SparseMatrix<double> &A, double eps, bool is_symmetric
) {
    return ensure_eigenvalue_lower_bound(A, eps, is_symmetric);
}
