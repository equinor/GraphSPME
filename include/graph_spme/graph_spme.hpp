// Main functions for GraphSPME
// License: GPL-3


#include <Eigen/Dense>
#include <Eigen/Sparse>

template <class T>
using Tvec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

using Dmat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using Dvec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using SpdMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
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
    Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic> &X, 
    int shrinkage_target=0,
    double inflation_factor=1.0
    )
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

    double lambda_hat;
    Dmat target_diagonal = Eigen::VectorXd::Ones(p);
    switch(shrinkage_target){
        case 0: {
            // common variance
            double nu = trS/ p;
            lambda_hat = (T_2N + T_1N*T_1N) / (
                n*T_2N + (p-n+1.0)/p * T_1N*T_1N
            );
            target_diagonal *= nu;
            break;
        }
        case 1: {
            // Identity
            lambda_hat = (T_2N + T_1N*T_1N) / (
                n*T_2N + T_1N*T_1N - (n-1)*(2.0 * T_1N - p)
                );
                // Target diagonal remains unit vector
            break;
        }
        case 2: {
            // Sample diagonal (variances)
            lambda_hat = (T_2N + T_1N * T_1N - 2.0 * T_3N) / (
                n * T_2N + T_1N * T_1N - (n + 1) * T_3N
                );
            target_diagonal = S.diagonal();
            break;
        }
        default:
            std::cerr << "Error: Shrinkage target is not 0, 1, or 2." << std::endl;
            break;
    }
    lambda_hat = std::max(0.0, std::min(lambda_hat, 1.0));


    S *= (1.0 - lambda_hat);
    S.diagonal() += lambda_hat * target_diagonal;

    // Inflate as a factor multiplied with BIC
    // BIC should give the order of sampling error
    S.diagonal() *= (1.0 + inflation_factor * 0.5 * p*std::log(p)/n);

    return S;
}

/*
 *  Sparse precision matrix inverse via sparse cholesky
 */
Dmat sparse_matrix_inverse(SpdMat &A)
{
    int p = A.rows();
    Eigen::SimplicialLLT<SpdMat> cholesky;
    cholesky.compute(A);
    SpdMat I(p, p);
    I.setIdentity();
    Dmat A_inv = cholesky.solve(I);
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
    bool symmetrization = true,
    int shrinkage_target=2,
    double inflation_factor=1.0
){
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
            cov_ml_est = cov_shrink_spd(xbi, shrinkage_target, inflation_factor);
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

/**
 * @brief Custom sparse matrix M to vector or triplets.
 * https://stackoverflow.com/a/51546701
 *
 * @param M sparse double matrix M.
 * @return std::vector of double triplets
 */
std::vector<dTriplet> to_triplets(SpdMat &M)
{
    std::vector<dTriplet> v;
    for (int i = 0; i < M.outerSize(); i++)
        for (typename SpdMat::InnerIterator it(M, i); it; ++it)
            v.emplace_back(it.row(), it.col(), it.value());
    return v;
}

/**
 * @brief Computes the trace of SQ where S is the normalized scatter matrix from data X.
 * The evaluation is computation and memory efficient given that:
 * - S is never explicitly calculated (memory efficient)
 * - Only iterates over non-zero elements of Q in trace formula.
 * Note that Symmetry in S and Q is not exploited.
 *
 * @param X nxp data matrix.
 * @param Q sparse SPD pxp matrix.
 * @return The trace of SQ.
 */
double trace_S_Q(Dmat &X, SpdMat &Prec)
{
    int n = X.rows();
    int p = X.cols();
    Dmat X_centered = X.rowwise() - X.colwise().mean();
    double trace_S_Prec = 0;
    int i = 0, j = 0;
    for (j = 0; j < p; j++)
    {
        for (typename SpdMat::InnerIterator it(Prec, j); it; ++it)
        {
            i = it.row();
            trace_S_Prec +=
                it.value() * X_centered.col(j).dot(X_centered.col(i)) / n;
        }
    }
    return trace_S_Prec;
}

/**
 * @brief Computes the Cholesky factor of a sparse matrix P with given permutation.
 *
 * @param P sparse SPD pxp matrix.
 * @param perm_indices vector of unique int 0<i<p-1 defining permutation of `P`.
 * @return The lower Cholesky factor L.
 */
SpdMat cholesky_factor(SpdMat &P, Tvec<int> perm_indices)
{
    // Ensure column-major
    P.makeCompressed();
    // Permute P
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> Perm(perm_indices);
    P = P.twistedBy(Perm);
    // Define cholesky solver to use natural ordering
    Eigen::SimplicialLLT<SpdMat, Eigen::Lower, Eigen::NaturalOrdering<int>> cholesky;
    cholesky.analyzePattern(P);
    cholesky.factorize(P);
    return cholesky.matrixL();
}

/**
 * @brief Retrieve origin matrix M from cholesky factor and permutation
 *
 * @param L the sparse lower triangular cholesky factor of P*M*P.T where P is defined from `perm_indices`
 * @param perm_indices vector of unique int 0<i<p-1 defining permutation of `P`.
 * @return SpTMat<double> SPD matrix M = P.T*L*L.T*P
 */
SpdMat chol_to_precision(SpdMat &L, Tvec<int> perm_indices)
{
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> Perm(perm_indices);
    SpdMat Prec = L * L.transpose();
    Prec = Prec.twistedBy(Perm.transpose());
    return Prec;
}

/**
 * @brief Log determinant of sparse SPD matrix `P`.
 * Computed efficiently using cholesky factor with known permutation defined by `perm_indices`
 *
 * @param P sparse SPD matrix.
 * @param perm_indices vector of unique int 0<i<p-1 defining permutation of `P`.
 * @return log determinant of `P`
 */
double logdet_sparse_spd(SpdMat &P, Tvec<int> perm_indices)
{
    SpdMat L = cholesky_factor(P, perm_indices);
    return 2.0 * L.diagonal().array().log().sum();
}

/**
 * @brief Approximate Minimum Degree (AMD) ordering.
 * The "state" of a Cholesky solver is in the optimized ordering.
 * Holding the permutation vector allows a purely function-based approach when working with the Cholesky decomposition.
 *
 * @param A sparse SPD pxp matrix.
 * @return Tvec<int> permutation indices, unique ints 0<i<p-1 defining permutation of `A` yielding AMD factorization.
 */
Tvec<int> compute_amd_ordering(SpdMat &A)
{
    Eigen::AMDOrdering<int> ordering;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> Perm(A.rows());
    ordering(A, Perm);
    Perm = Perm.transpose();
    return Perm.indices(); // equals Cholesky AMD cholesky.permutationP().indices()
}

/**
 * @brief Negative log-likelihood of GMRF relevant to precision.
 *
 * @param X nxp data matrix.
 * @param Prec pxp spd precision matrix.
 * @param perm_indices vector of unique int 0<i<p-1 defining AMD permutation of `Prec`.
 * @return negative log-likelihood.
 */
double dmrf(Dmat &X, SpdMat &Prec, Tvec<int> perm_indices)
{
    // Ensure column-major format
    Prec.makeCompressed();
    double trace_S_Prec = trace_S_Q(X, Prec);
    double prec_log_det = logdet_sparse_spd(Prec, perm_indices);
    return 0.5 * (trace_S_Prec - prec_log_det);
}

/**
 * @brief Negative log-likelihood of GMRF relevant to cholesky factor of precision.
 *
 * @param X nxp data matrix.
 * @param L pxp lower-triangular cholesky factor of permuted precision matrix.
 * @param perm_indices vector of unique int 0<i<p-1 defining AMD for `L`.
 * @return negative log-likelihood.
 */
double dmrfL(Dmat &X, SpdMat &L, Tvec<int> perm_indices)
{
    // Ensure L in column-major format
    L.makeCompressed();
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> Perm(perm_indices);
    SpdMat Prec = L * L.transpose();
    Prec = Prec.twistedBy(Perm.transpose()); // reverse L = chol(P * Lambda * P.T)
    double trace_S_Prec = trace_S_Q(X, Prec);
    double prec_log_det = 2.0 * L.diagonal().array().log().sum();
    return 0.5 * (trace_S_Prec - prec_log_det);
}

/*
 * Returns the nearest symmetric semidefinite matrix of A in Frobenius norm.
 * The result is from
 * Higham NJ (1988). “Computing a nearest symmetric positive semidefinite matrix.”
 * A_opt=0.5*(H+U),
 * H=0.5*(A+A.t) nearest symmetric matrix
 * U: polar factor in polar decomposition H=PU
 * The proof of A_opt also shows that
 * A_opt=QD+Q.t,
 * where QDQ.t is the eigen/spectral decomposition of H and D+_ii=max(eps>0, D_ii)
 * A is symmetrized if is_symmetric evaluates to false.
 */
SpdMat ensure_eigenvalue_lower_bound(
    SpdMat &A, double eps, bool is_symmetric)
{
    // Closest symmetrization
    if (!is_symmetric)
    {
        ensure_symmetry(A);
    }
    // Eigendecomposition
    Eigen::MatrixXd A_dense(A);
    Eigen::EigenSolver<Eigen::MatrixXd> eigen_decomposition(A_dense);
    Eigen::VectorXd eigenvalues = eigen_decomposition.eigenvalues().array().real();
    Eigen::MatrixXd D = eigenvalues.asDiagonal();
    Eigen::MatrixXd Q = eigen_decomposition.eigenvectors().array().real();
    // Modify eigenvalues
    for (int i = 0; i < D.rows(); i++)
    {
        D(i, i) = std::max(eps, D(i, i));
    }
    // Build and return closest symmetric semidefinite in FB-norm
    Eigen::MatrixXd QDpQt = Q * D;
    QDpQt *= Q.transpose();
    // Pick out sparse elements -- whatever non-zero is in zero-elements is numerical error
    int nz = A.nonZeros();
    std::vector<dTriplet> A_triplets = to_triplets(A);
    for (int i = 0; i < nz; i++)
    {
        dTriplet tri = dTriplet(
            A_triplets[i].row(),
            A_triplets[i].col(),
            QDpQt(A_triplets[i].row(), A_triplets[i].col()));
        A_triplets[i] = tri;
    }
    A.setFromTriplets(A_triplets.begin(), A_triplets.end());
    return A;
}

/////// hand-coded derivatives ////////
/**
 * @brief Matrix derivative of tr(S*Prec)/n w.r.t. Prec
 *
 * @param X nxp data matrix
 * @return matrix derivative
 */
Dmat dtrace_S_Prec(Dmat &X)
{
    int n = X.rows();
    int p = X.cols();
    Dmat X_centered = X.rowwise() - X.colwise().mean();
    Dmat cov_sample = X_centered.transpose() * X_centered;
    cov_sample = cov_sample / n;
    // Multiply every element of the matrix by 2 except the diagonal elements
    // Corresponds to multiplication with Duplication matrix and due to symmetry of Prec
    // https://en.wikipedia.org/wiki/Duplication_and_elimination_matrices
    for (int i = 0; i < p; i++)
    {
        for (int j = 0; j < p; j++)
        {
            if (i != j)
            {
                cov_sample(i, j) *= 2.0;
            }
        }
    }
    return cov_sample;
}

/**
 * @brief Matrix derivative of logdet(Prec) w.r.t. Prec
 *
 * @param Prec pxp spd precision matrix
 * @return matrix derivative
 */
Dmat dprec_log_det(SpdMat &Prec)
{
    Dmat res = sparse_matrix_inverse(Prec);
    // Multiply every element of the matrix by 2 except the diagonal elements
    // Corresponds to multiplication with Duplication matrix and due to symmetry of Prec
    // https://en.wikipedia.org/wiki/Duplication_and_elimination_matrices
    for (int i = 0; i < res.rows(); ++i)
    {
        for (int j = 0; j < res.cols(); ++j)
        {
            if (i != j)
            {
                res(i, j) *= 2.0;
            }
        }
    }
    return res;
}

/**
 * @brief Gradient of negative log-likelihood of GMRF w.r.t. precision.
 * Calculations will be done for precision in sparse column major ordering.
 *
 * @param X nxp data matrix.
 * @param Prec pxp spd precision matrix.
 * @param grad_elements_pick pxp spd subset of `Prec` for which to return the gradient.
 * @return negative log-likelihood.
 */
Dmat dmrf_grad(Dmat &X, SpdMat &Prec, SpdMat &grad_elements_pick)
{
    grad_elements_pick = grad_elements_pick.triangularView<Eigen::Lower>();
    int nparameters = grad_elements_pick.nonZeros();
    std::vector<dTriplet> grad_elements_pick_triplets = to_triplets(grad_elements_pick);
    Dmat grad_mat = 0.5 * (dtrace_S_Prec(X) - dprec_log_det(Prec));
    Dvec grad(nparameters);
    for (int i = 0; i < nparameters; i++)
    {
        grad[i] = grad_mat(
            grad_elements_pick_triplets[i].row(),
            grad_elements_pick_triplets[i].col());
    }
    return grad;
}

/**
 * @brief Hessian of negative log-likelihood of GMRF w.r.t. precision.
 * Calculations will be done for precision in sparse column major ordering.
 *
 * @param Prec pxp spd precision matrix.
 * @param grad_elements_pick pxp spd subset of `Prec` for which to return the hessian.
 * @return negative log-likelihood.
 */
Dmat dmrf_hess(SpdMat &Prec, SpdMat &grad_elements_pick)
{
    // grad = -0.5 * (2Xinv - elementwisemul(Xinv, I))
    // grad_elements_pick should already be lower triangular, but just in case
    grad_elements_pick = grad_elements_pick.triangularView<Eigen::Lower>();
    int nparameters = grad_elements_pick.nonZeros();
    std::vector<dTriplet> grad_elements_pick_triplets = to_triplets(grad_elements_pick);
    Dmat hess(nparameters, nparameters);
    int i, j, k, l;
    // find symmetric(!) matrix derivative of X_inv and then apply it to grad
    double dX_inv;
    Dmat cov = sparse_matrix_inverse(Prec);
    for (int m = 0; m < nparameters; m++)
    {
        i = grad_elements_pick_triplets[m].row();
        j = grad_elements_pick_triplets[m].col();
        for (int n = 0; n < nparameters; n++)
        {
            k = grad_elements_pick_triplets[n].row();
            l = grad_elements_pick_triplets[n].col();
            dX_inv = -cov(k, i) * cov(j, l);
            dX_inv += -cov(k, j) * cov(i, l);
            if (i == j)
            {
                dX_inv -= -cov(k, i) * cov(j, l);
            }
            // apply to grad
            hess(m, n) = k != l ? 2.0 * dX_inv : dX_inv;
            hess(m, n) = -0.5 * hess(m, n);
        }
    }
    return hess;
}

/**
 * @brief Gradient of negative log-likelihood of GMRF w.r.t. lower cholesky factor of precision.
 * Calculations and all vectorizations will be done for cholesky factor in sparse column major ordering.
 *
 * @param X nxp data matrix.
 * @param L pxp spd cholesky factor of precision matrix.
 * @param grad_elements_pick pxp spd subset of `L` for which to return the gradient.
 * @param perm_indices vector of unique int 0<i<p-1 defining AMD permutation of `precision matrix`.
 * @return gradient of Gaussian negative log-likelihood w.r.t. elements in L.
 */
Dmat dmrfL_grad(Dmat &X, SpdMat &L, SpdMat &grad_elements_pick, Tvec<int> perm_indices)
{
    grad_elements_pick = grad_elements_pick.triangularView<Eigen::Lower>();
    int nparameters = grad_elements_pick.nonZeros();
    std::vector<dTriplet> grad_elements_pick_triplets = to_triplets(grad_elements_pick);
    int n = X.rows();
    Dmat X_centered = X.rowwise() - X.colwise().mean();
    Dmat cov_sample = X_centered.transpose() * X_centered / n;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> Perm(perm_indices);
    cov_sample = Perm * cov_sample;
    cov_sample = cov_sample * Perm.transpose();
    Dmat d_SPrec_dL = 2.0 * cov_sample * L;
    Eigen::VectorXd L_diag_inv = 2.0 / L.diagonal().array();
    Dmat d_logdetPrec_dL = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(
        L_diag_inv);
    Dmat grad_mat = 0.5 * (d_SPrec_dL - d_logdetPrec_dL);
    Tvec<double> grad(nparameters);
    for (int i = 0; i < nparameters; i++)
    {
        grad[i] = grad_mat(
            grad_elements_pick_triplets[i].row(),
            grad_elements_pick_triplets[i].col());
    }
    return grad;
}

/**
 * @brief Hessian of negative log-likelihood of GMRF w.r.t. lower cholesky factor of precision.
 * Calculations and all vectorizations will be done for cholesky factor in sparse column major ordering.
 *
 * @param X nxp data matrix.
 * @param L pxp spd cholesky factor of precision matrix.
 * @param grad_elements_pick pxp spd subset of `L` for which to return the hessian.
 * @param perm_indices vector of unique int 0<i<p-1 defining AMD permutation of `precision matrix`.
 * @return Hessian of Gaussian negative log-likelihood w.r.t. elements in L.
 */
Dmat dmrfL_hess(Dmat &X, SpdMat &L, SpdMat &grad_elements_pick, Tvec<int> perm_indices)
{
    grad_elements_pick = grad_elements_pick.triangularView<Eigen::Lower>();
    int nparameters = grad_elements_pick.nonZeros();
    std::vector<dTriplet> grad_elements_pick_triplets = to_triplets(grad_elements_pick);
    int n = X.rows();
    Dmat X_centered = X.rowwise() - X.colwise().mean();
    Dmat cov_sample = X_centered.transpose() * X_centered / n;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> Perm(perm_indices);
    cov_sample = Perm * cov_sample;
    cov_sample = cov_sample * Perm.transpose();
    Dvec L_diag_square = L.diagonal().array().pow(2.0);
    Dmat hess(nparameters, nparameters);
    hess.setZero();
    int i, j, k, l;
    for (int m = 0; m < nparameters; m++)
    {
        i = grad_elements_pick_triplets[m].row();
        j = grad_elements_pick_triplets[m].col();
        for (int n = 0; n < nparameters; n++)
        {
            k = grad_elements_pick_triplets[n].row();
            l = grad_elements_pick_triplets[n].col();
            if (i == k && j == l)
            {
                hess(m, n) = cov_sample(i, j);
                if (i == j)
                {
                    hess(m, n) += 1.0 / L_diag_square[i];
                }
            }
        }
    }
    return hess;
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
    Tvec<int> perm_indices = compute_amd_ordering(Prec);
    return dmrf(X, Prec, perm_indices);
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
