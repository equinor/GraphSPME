// Main functions for GraphSPME
// License: GPL-3
#include <Eigen/Dense>
#include <Eigen/Sparse>

using Dmat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using SpdMat = Eigen::SparseMatrix<double>;
// using SpdMatMap = Eigen::MappedSparseMatrix<double>;
using dTriplet = Eigen::Triplet<double>;

/*
 * Returns matrix of non-zero values for precision matrix
 * for a distribution having a Markov property of order `markov_order`
 *
 * @param Neighbours Sparse matrix of ones indicating neighbours on a graph
 * @param markov_order int order of the Markov property
 * @return sparse matrix with non-zero elements corresponding to the precision
 */
Eigen::SparseMatrix<double> get_precision_nonzero(
    Eigen::SparseMatrix<double> Neighbours,
    int markov_order)
{
    // Return identity matrix if order zero
    if (markov_order == 0)
    {
        int p = Neighbours.rows();
        Eigen::SparseMatrix<double> I(p, p);
        I.setIdentity();
        return I;
    }
    // Propagate the information to neighbours through multiplication
    Eigen::SparseMatrix<double> G = Neighbours;
    for (int order = 1; order < markov_order; order++)
    {
        G = G * Neighbours;
    }
    // Reset all non-zero values to ones
    for (int k = 0; k < G.outerSize(); ++k)
    {
        for (SpdMat::InnerIterator it(G, k); it; ++it)
        {
            it.valueRef() = 1.0;
        }
    }
    return G;
}

/*
 * Returns matrix Bi so that Bi*wi1 = wi,
 * wi: column i of precision matrix
 * wi1: non-zero elements of wi
 * Z: sparse positional matrix of non-zero elements of precision
 * j: column
 */
Eigen::SparseMatrix<double> create_bi(
    Eigen::SparseMatrix<double> &Z,
    int j)
{
    int p = Z.cols();
    // Iterate to find non-zero elements of Z[,i]
    int si = 0;
    std::vector<int> row_values;
    for (SpdMat::InnerIterator it(Z, j); it; ++it)
    {
        si += it.value();
        row_values.push_back(it.row());
    }
    // Use triplets to initialize block I_si at start
    std::vector<dTriplet> sparse_mat_triplet(si);
    for (int i = 0; i < si; i++)
    {
        sparse_mat_triplet[i] = dTriplet(row_values[i], i, 1.0);
    }
    SpdMat Bi(p, si);
    Bi.setFromTriplets(sparse_mat_triplet.begin(), sparse_mat_triplet.end());
    return Bi;
}

/*
 * The maximum likelihood covariance estimate
 */
Dmat cov_ml(Dmat &X)
{
    // Likelihood estimate of covariance
    Dmat centered = X.rowwise() - X.colwise().mean();
    Dmat cov = (centered.adjoint() * centered) / double(X.rows() - 1);
    return cov;
}

/*
 * Covariance shrinkage estimate as specified in Touloumis (2015)
 */
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cov_shrink_spd(
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &X)
{
    int n = X.rows();
    int p = X.cols();

    // Calculate T_1N and T_2N
    // Calculations may be done without calculating S, see appendix C of Touloumis (2015)
    // Choice depends on p and n
    Dmat centered = X.rowwise() - X.colwise().mean();
    Dmat S = (centered.adjoint() * centered) / double(n - 1.0);
    double trS = S.trace();
    double trS2 = S.squaredNorm(); // tr(A*B) = sum_i,j a_ij * b_ij
    double q = centered.rowwise().squaredNorm().squaredNorm() / (n - 1.0);
    double T_1N = trS;
    double T_2N = (n - 1.0) * ((n - 1.0) * (n - 2.0) * trS2 + trS * trS - n * q) / (n * (n - 2) * (n - 3));

    // Calculate T_3N
    Dmat x_col_sum = X.colwise().sum();
    Dmat Sum1(1, p), Sum21(1, p), Sum22(1, p);
    Sum1.setZero();
    Sum21.setZero();
    Sum22.setZero();
    Dmat x_square = X.cwiseProduct(X);
    Dmat x_cube = x_square.cwiseProduct(X);
    Dmat x_minus_i = X.colwise().sum();
    Dmat x_square_minus_i = x_square.colwise().sum(); // Notation of code for paper
    Dmat x_cube_minus_i = x_cube.colwise().sum();     // Notation of code for paper
    double Y_3N = 0.0;
    for (int i = 0; i < n; i++)
    {
        x_minus_i -= X.row(i);
        x_square_minus_i -= x_square.row(i);
        x_cube_minus_i -= x_cube.row(i);
        Sum1 += X.row(i).cwiseProduct(x_minus_i);
        Sum21 += x_cube.row(i).cwiseProduct(x_minus_i);
        Sum22 += x_cube_minus_i.cwiseProduct(X.row(i));
        Y_3N += x_square_minus_i.cwiseProduct(x_square.row(i)).sum();
    }
    double Y_7N = 2 * (Sum1.cwiseProduct(x_square.colwise().sum()).sum() - (Sum21 + Sum22).sum());
    double Y_8N = 4 * (Sum1.squaredNorm() - Y_3N - Y_7N);
    Y_3N = 2 * Y_3N / (n * (n - 1));
    Y_7N = Y_7N / (n * (n - 1) * (n - 2));
    Y_8N = Y_8N / (n * (n - 1) * (n - 2) * (n - 3));
    double T_3N = Y_3N - 2 * Y_7N + Y_8N;

    // Calculate shrinkage factor
    // Target is diag(S)
    // Avoid target selection as recommended in paper
    // empirical study: diag(S) works best
    double lambda_hat_D = (T_2N + T_1N * T_1N - 2.0 * T_3N) / (n * T_2N + T_1N * T_1N - (n + 1) * T_3N);
    lambda_hat_D = std::max(0.0, std::min(lambda_hat_D, 1.0));

    // Modify covariance estimate
    double lambda_hat = lambda_hat_D;
    Dmat target_diagonal = S.diagonal();
    S *= (1.0 - lambda_hat);
    S.diagonal() += lambda_hat * target_diagonal;
    return S;
}

/*
 *  Sparse precision matrix inverse
 *  Employs cojugate gradient
 *  Recomended by http://eigen.tuxfamily.org/dox-devel/group__TopicSparseSystems.html
 */
Dmat sparse_matrix_inverse(SpdMat &A)
{
    int p = A.rows();
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    Eigen::SparseMatrix<double> I(p, p);
    I.setIdentity();
    auto A_inv = solver.solve(I);
    return A_inv;
}

/*
 * Ensure symmetry of matrix
 */
void ensure_symmetry(SpdMat &A)
{
    SpdMat At = A.transpose();
    A = A + At;
    A *= 0.5;
}

/*
 * Graphical sparse precision matrix estimation
 * as defined in Le (2021)
 * and possibilities of using covariance shrinkage from Lunde etal (2022?)
 */
Eigen::SparseMatrix<double> prec_sparse(
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &X,
    Eigen::SparseMatrix<double> &Graph,
    int markov_order = 1,
    bool cov_shrinkage = true,
    bool symmetrization = true)
{
    int p = X.cols();
    int values_set = 0;
    int si;
    SpdMat Ip(p, p), Prec(p, p);
    Ip.setIdentity();
    Prec.setZero();
    Eigen::SparseMatrix<double> Z = get_precision_nonzero(Graph, markov_order);
    std::vector<dTriplet> prec_mat_triplet(Z.nonZeros());
    for (int j = 0; j < p; j++)
    {
        SpdMat Bi = create_bi(Z, j);
        SpdMat Bi_trans = Bi.transpose();
        si = Bi.cols();
        Dmat xbi = X * Bi;
        Dmat cov_ml_est(si, si);
        if (cov_shrinkage)
        {
            cov_ml_est = cov_shrink_spd(xbi);
        }
        else
        {
            cov_ml_est = cov_ml(xbi);
        }
        auto wi1 = cov_ml_est.inverse() * (Bi_trans * Ip.col(j));
        for (int k = 0; k < Bi.outerSize(); ++k)
        {
            for (SpdMat::InnerIterator it(Bi, k); it; ++it)
            {
                prec_mat_triplet[values_set] = dTriplet(it.row(), j, wi1[it.col()]);
                values_set++;
            }
        }
    }
    Prec.setFromTriplets(prec_mat_triplet.begin(), prec_mat_triplet.end());
    if (symmetrization)
    {
        ensure_symmetry(Prec);
    }
    return Prec;
}

/*
 ***** fast likelihood based functions and derivatives for estimation  of precision *****
 */
#include "../adept/adept.h"
#include "../adept/adept_source.h"
#include <math.h>
using adept::adouble;
typedef adept::adouble adtype;

template <class T>
using Tmat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <class T>
using Tvec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <class T>
using SpTMat = Eigen::SparseMatrix<T>;

template <class T>
using TTriplet = Eigen::Triplet<T>;

// NOTE: I would really like to avoid this. Ask Colleagues
// adept header warns of this, I think
// eigen sparse chol can not find sqrt and log functions in adept namespace without this
namespace adept
{
    adouble sqrt(adouble x)
    {
        return Sqrt<adouble>(x);
    }

    adouble log(adouble x)
    {
        return Log<adouble>(x);
    }
}

/*
 * Custom sparse matrix M to vector or triplets
 * https://stackoverflow.com/a/51546701
 */
template <class T>
std::vector<TTriplet<T>> to_triplets(SpTMat<T> &M)
{
    std::vector<TTriplet<T>> v;
    for (int i = 0; i < M.outerSize(); i++)
        for (typename SpTMat<T>::InnerIterator it(M, i); it; ++it)
            v.emplace_back(it.row(), it.col(), it.value());
    return v;
}

/*
 * Returns trace of matrix product S*Q.
 * Assuming S is symmetric pxp and Q sparse symmetric.
 * Trace is computation and memory efficient given assumptions above.
 * - S is never explicitly calculated (memory efficient)
 * - Only iterates over non-zero elements of Q in trace formula.
 * X: data-matrix nxp. S = (X-mean(X))^T * (X-mean(X))
 * Q: sparse matrix Q
 */
template <class T>
T trace_S_Q(Tmat<double> &X, SpTMat<T> &Prec)
{
    int n = X.rows();
    int p = X.cols();
    Tmat<double> X_centered = X.rowwise() - X.colwise().mean();
    T trace_S_Prec = 0;
    int i = 0, j = 0;
    for (j = 0; j < p; j++)
    {
        for (typename SpTMat<T>::InnerIterator it(Prec, j); it; ++it)
        {
            i = it.row();
            trace_S_Prec +=
                it.value() * X_centered.col(j).dot(X_centered.col(i)) / n;
        }
    }
    return trace_S_Prec;
}

template <class T>
T preclogdet(SpTMat<T> &P)
{
    Eigen::SimplicialLLT<SpTMat<T>> cholesky;
    cholesky.analyzePattern(P);
    cholesky.factorize(P);
    SpTMat<T> L = cholesky.matrixL();
    return 2.0 * L.diagonal().array().log().sum();
}

Tvec<int> computeAMDOrdering(SpTMat<double> &A)
{
    Eigen::AMDOrdering<int> ordering;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> Perm(A.rows());
    ordering(A, Perm);
    return Perm.indices();
}

int sparseCholeskyNonzeroElementsCount(SpTMat<double> &A)
{
    Eigen::SimplicialLLT<SpTMat<double>> cholesky;
    cholesky.analyzePattern(A);
    cholesky.factorize(A);
    SpTMat<double> L = cholesky.matrixL();
    return L.nonZeros();
}

SpTMat<double> sparseCholeskyGraph(SpTMat<double> &A)
{
    Eigen::SimplicialLLT<SpTMat<double>> cholesky;
    cholesky.analyzePattern(A);
    cholesky.factorize(A);
    SpTMat<double> L = cholesky.matrixL();
    for (int k = 0; k < L.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
            it.valueRef() = 1.0;
    return L;
}

/*
 * Relevant part of negative-log density of Markov random field
 * Only relevant part returned:
 * 0.5 * (tr(SQ) - log(det(Q)))
 */
template <class T>
T dmrf(Tmat<double> &X, SpTMat<T> &Prec)
{
    T trace_S_Prec = trace_S_Q<T>(X, Prec);
    T prec_log_det = preclogdet(Prec);
    T nll = 0.5 * (trace_S_Prec - prec_log_det);
    return nll;
}

template <class T>
T dmrfL(Tmat<double> &X, SpTMat<T> &L, Tvec<int> perm_indices)
{
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> Perm(perm_indices);
    SpTMat<T> Prec = L * L.transpose();
    Prec = Prec.twistedBy(Perm); // reverse L = chol(P Lambda P.T)
    T trace_S_Prec = trace_S_Q<T>(X, Prec);
    T prec_log_det = 2.0 * L.diagonal().array().log().sum();
    return 0.5 * (trace_S_Prec - prec_log_det);
}

/*
 * Interface to dmrf with doubles
 */
double dmrf(Tmat<double> &X, SpTMat<double> &Prec)
{
    return dmrf<double>(X, Prec);
}

double dmrfL(Tmat<double> &X, SpTMat<double> &L, Tvec<int> perm_indices)
{
    return dmrfL<double>(X, L, perm_indices);
}

Tvec<double> ddmrf(Tmat<double> &X, SpTMat<double> &Prec)
{

    int nz = Prec.nonZeros();
    Prec = Prec.triangularView<Eigen::Lower>();
    int nparameters = Prec.nonZeros();
    int p = Prec.cols();

    // Start stack
    adept::Stack stack;

    // create args-vector to compute gradient for
    adouble args_ad[nparameters];
    std::vector<TTriplet<double>> Prec_lower_tri_dtriplets = to_triplets<double>(Prec);
    for (int i = 0; i < nparameters; i++)
    {
        args_ad[i] = Prec_lower_tri_dtriplets[i].value();
    }

    // Clean derivative information
    stack.new_recording();

    // Init ad version of Prec
    std::vector<TTriplet<adouble>> Prec_adtriplets(nz);
    TTriplet<double> tri;
    for (int i = 0; i < nparameters; i++)
    {
        tri = Prec_lower_tri_dtriplets[i];
        Prec_adtriplets[i] = TTriplet<adouble>(
            tri.row(), tri.col(), args_ad[i]);
        if (tri.row() != tri.col())
        {
            // not diagonal, add symmetric element in prec
            Prec_adtriplets[i] = TTriplet<adouble>(
                tri.col(), tri.row(), args_ad[i]);
        }
    }
    SpTMat<adouble> Prec_ad(p, p);
    Prec_ad.setFromTriplets(Prec_adtriplets.begin(), Prec_adtriplets.end());

    // Compute objective function
    adouble nll = dmrf<adouble>(X, Prec_ad);

    // Compute gradient
    nll.set_gradient(1.0);
    stack.compute_adjoint();

    // Extract gradient elements
    Tvec<double> grad(nparameters);
    for (int i = 0; i < nparameters; i++)
    {
        grad[i] = args_ad[i].get_gradient();
    }
    return grad;
}

/*
 * Returns gradient of dmrf w.r.t. vectorized (row-by-row) L
 * L: sparse col-major matrix, left Cholesky factor of precision matrix.
 */
Tvec<double> ddmrfL(Tmat<double> &X, SpTMat<double> &L, Tvec<int> perm_indices)
{
    int nz = L.nonZeros();
    int p = L.cols();

    // Start stack
    adept::Stack stack;

    // create args-vector to compute gradient for
    adouble args_ad[nz];
    std::vector<TTriplet<double>> L_dtriplets = to_triplets<double>(L);
    for (int i = 0; i < nz; i++)
    {
        args_ad[i] = L_dtriplets[i].value();
    }

    // Clean derivative information
    stack.new_recording();

    // Init ad version of L
    std::vector<TTriplet<adouble>> L_adtriplets(nz);
    TTriplet<double> tri;
    for (int i = 0; i < nz; i++)
    {
        tri = L_dtriplets[i];
        L_adtriplets[i] = TTriplet<adouble>(
            tri.row(), tri.col(), args_ad[i]);
    }
    SpTMat<adouble> L_ad(p, p);
    L_ad.setFromTriplets(L_adtriplets.begin(), L_adtriplets.end());

    // Compute objective function
    adouble nll = dmrfL<adouble>(X, L, perm_indices);

    // Compute gradient
    nll.set_gradient(1.0);
    stack.compute_adjoint();

    // Extract gradient elements
    Tvec<double> grad(nz);
    for (int i = 0; i < nz; i++)
    {
        grad[i] = args_ad[i].get_gradient();
    }
    return grad;
}

///////////////////////////////////

/*
 * Gaussian negative log-likelihood
 * Relevant part for precision matrix estimation
 */
double prec_nll(
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &X,
    SpdMat &Prec)
{
    return dmrf(X, Prec);
}

/*
 * AIC with relevant part of Gaussian likelihood
 */
double prec_aic(
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &X,
    SpdMat &Prec)
{
    double nll = prec_nll(X, Prec);
    // AIC penalization factor: The number of free parameters in the precision
    int p = Prec.rows();
    int n = X.rows();
    int nz = Prec.nonZeros();
    double penalty = (nz + p) / 2.0 / n;
    // aic
    double aic = nll + penalty;
    return aic;
}
