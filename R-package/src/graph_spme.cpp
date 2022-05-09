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
 *  Employs cojugate gradient
 *  Recomended by http://eigen.tuxfamily.org/dox-devel/group__TopicSparseSystems.html
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
        bool cov_shrinkage=true
){
    return prec_sparse(x, Neighbours, markov_order, cov_shrinkage);
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
