// Includes all header files for GraphSPME
// License: GPL-3

#ifndef __GRAPHSPME_HPP_INCLUDED__
#define __GRAPHSPME_HPP_INCLUDED__

#include <RcppEigen.h>

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins("cpp11")]]

// Enables Eigen
// // [[Rcpp::depends(RcppEigen)]]

template <class T>
using Tvec = Eigen::Matrix<T,Eigen::Dynamic,1>;

template <class T>
using Tmat = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;

template<class T>
using Tavec = Eigen::Array<T,Eigen::Dynamic,1>; 

template<class T>
using SpTMat = Eigen::SparseMatrix<T>;
template<class T>
using SpTMatMap = Eigen::MappedSparseMatrix<T>;

using SpdMat = Eigen::SparseMatrix<double>;
using SpdMatMap = Eigen::MappedSparseMatrix<double>;
using SpiMat = Eigen::SparseMatrix<int>;
using SpiMatMap = Eigen::MappedSparseMatrix<double>;

#endif // __GRAPHSPME_HPP_INCLUDED__