// Main functions for GraphSPME
// License: GPL-3
#include <Eigen/Dense>
#include <Eigen/Sparse>



using Dmat = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>;
using SpdMat = Eigen::SparseMatrix<double>;
// using SpdMatMap = Eigen::MappedSparseMatrix<double>;
using dTriplet = Eigen::Triplet<double>;


/*
 * Returns matrix Bi so that Bi*wi1 = wi, 
 * wi: column i of precision matrix
 * wi1: non-zero elements of wi
 * Z: sparse positional matrix of non-zero elements of precision
 * j: column
 */
Eigen::SparseMatrix<double> create_bi(
        Eigen::SparseMatrix<double>& Z, 
        int j
    ){
    int p = Z.cols();
    // Iterate to find non-zero elements of Z[,i]
    int si = 0;
    std::vector<int> row_values;
    for(SpdMat::InnerIterator it(Z,j); it; ++it) {
        si += it.value();
        row_values.push_back(it.row());
    }
    // Use triplets to initialize block I_si at start
    std::vector<dTriplet> sparse_mat_triplet(si);
    for(int i=0; i<si; i++){
        sparse_mat_triplet[i] = dTriplet(row_values[i],i,1.0);
    }
    SpdMat Bi(p,si);
    Bi.setFromTriplets(sparse_mat_triplet.begin(), sparse_mat_triplet.end());
    return Bi;
}

/*
 * The maximum likelihood covariance estimate
 */
Dmat cov_ml(Dmat& x){
    // Likelihood estimate of covariance
    Dmat centered = x.rowwise() - x.colwise().mean();
    Dmat cov = (centered.adjoint() * centered) / double(x.rows() - 1);
    return cov;
}

/*
 * Covariance shrinkage estimate as specified in Touloumis (2015)
 */
Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> cov_shrink_spd(
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& x
    ){
    int n = x.rows();
    int p = x.cols();
    
    // Calculate T_1N and T_2N
    // Calculations may be done without calculating S, see appendix C of Touloumis (2015)
    // Choice depends on p and n
    Dmat centered = x.rowwise() - x.colwise().mean();
    Dmat S = (centered.adjoint() * centered) / double(n-1.0);
    double trS = S.trace();
    double trS2 = S.squaredNorm(); // tr(A*B) = sum_i,j a_ij * b_ij
    double q = centered.rowwise().squaredNorm().squaredNorm() / (n-1.0);
    double T_1N = trS;
    double T_2N = (n-1.0)*((n-1.0)*(n-2.0)*trS2 + trS*trS - n*q) / (n*(n-2)*(n-3));
    
    // Calculate T_3N
    Dmat x_col_sum = x.colwise().sum();
    Dmat Sum1(1,p), Sum21(1,p), Sum22(1,p);
    Sum1.setZero(); 
    Sum21.setZero(); 
    Sum22.setZero();
    Dmat x_square = x.cwiseProduct(x);
    Dmat x_cube = x_square.cwiseProduct(x);
    Dmat x_minus_i = x.colwise().sum();
    Dmat x_square_minus_i = x_square.colwise().sum(); // Notation of code for paper
    Dmat x_cube_minus_i = x_cube.colwise().sum(); // Notation of code for paper
    double Y_3N=0.0;
    for(int i=0; i<n; i++){
        x_minus_i -= x.row(i);
        x_square_minus_i -= x_square.row(i);
        x_cube_minus_i -= x_cube.row(i);
        Sum1 += x.row(i).cwiseProduct(x_minus_i);
        Sum21 += x_cube.row(i).cwiseProduct(x_minus_i);
        Sum22 += x_cube_minus_i.cwiseProduct(x.row(i));
        Y_3N += x_square_minus_i.cwiseProduct(x_square.row(i)).sum();
    }
    double Y_7N = 2 * (Sum1.cwiseProduct(x_square.colwise().sum()).sum() - (Sum21+Sum22).sum());
    double Y_8N = 4 * (Sum1.squaredNorm() - Y_3N - Y_7N);
    Y_3N = 2*Y_3N/(n*(n-1));
    Y_7N = Y_7N/(n*(n-1)*(n-2));
    Y_8N = Y_8N/(n*(n-1)*(n-2)*(n-3));
    double T_3N = Y_3N - 2 * Y_7N + Y_8N;
    
    // Calculate shrinkage factor
    // Target is diag(S)
    // Avoid target selection as recommended in paper
    // empirical study: diag(S) works best
    double lambda_hat_D = (T_2N + T_1N*T_1N - 2.0*T_3N) / (n*T_2N+T_1N*T_1N-(n+1)*T_3N);
    lambda_hat_D = std::max(0.0, std::min(lambda_hat_D,1.0));
    
    // Modify covariance estimate
    double lambda_hat= lambda_hat_D;
    Dmat target_diagonal = S.diagonal();
    S *= (1.0-lambda_hat);
    S.diagonal() += lambda_hat * target_diagonal;
    return S;
}

/*
 *  Sparse precision matrix inverse
 *  Employs cojugate gradient
 *  Recomended by http://eigen.tuxfamily.org/dox-devel/group__TopicSparseSystems.html
 */
Dmat sparse_matrix_inverse(SpdMat& A){
    int p = A.rows();
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double> > solver;
    solver.compute(A);
    Eigen::SparseMatrix<double> I(p,p);
    I.setIdentity();
    auto A_inv = solver.solve(I);
    return A_inv;
}

/*
 * Graphical sparse precision matrix estimation
 * as defined in Le (2021)
 * and possibilities of using covariance shrinkage from Lunde etal (2022?)
 */
Eigen::SparseMatrix<double> prec_sparse(
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& x,
    Eigen::SparseMatrix<double>& Z,
    bool cov_shrinkage=true
){
    int n = x.rows();
    int p = x.cols();
    int values_set = 0;
    int si;
    SpdMat Ip(p,p), Prec(p,p);
    Ip.setIdentity();
    Prec.setZero();
    std::vector<dTriplet> prec_mat_triplet(Z.nonZeros());
    for(int j=0; j<p; j++){
        SpdMat Bi = create_bi(Z,j);
        SpdMat Bi_trans = Bi.transpose();
        si = Bi.cols();
        Dmat xbi = x * Bi;
        Dmat cov_ml_est(si,si);
        if(cov_shrinkage){
            cov_ml_est = cov_shrink_spd(xbi);
        }else{
            cov_ml_est = cov_ml(xbi);   
        }
        auto wi1 = cov_ml_est.inverse() * (Bi_trans * Ip.col(j));
        for (int k=0; k < Bi.outerSize(); ++k)
        {
            for(SpdMat::InnerIterator it(Bi,k); it; ++it) {
                prec_mat_triplet[values_set] = dTriplet(it.row(), j, wi1[it.col()]);
                values_set++;
            }
        }
    }
    Prec.setFromTriplets(prec_mat_triplet.begin(), prec_mat_triplet.end());
    return Prec;
}
