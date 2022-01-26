# GraphSPME

**High dimensional precision matrix estimation with a known graphical structure**

- [x] Works in very high dimensions
- [x] Non-parametric
- [x] Asymptotic regularization 🖖
- [x] Lightning fast
- [x] Both Python and R

_Note: Still work in progress. Waiting for Py_

## Installation

**R**: Install the development version from GitHub
```r
devtools::install_github("Blunde1/GraphSPME/GraphSPME")
```

## Example code and documentation
Simulate a zero-mean AR1 process with a known graphical structure:
```r
library(Matrix)
# Zero mean AR1
rar1 <- function(n, psi){
    x <- numeric(n)
    x[1] <- rnorm(1, 0, 1/sqrt(1-psi^2))
    for(i in 2:n){
        x[i] <- psi*x[i-1] + rnorm(1)
    }
    return(x)
}
# Simulate data
set.seed(123)
p <- 5
psi <- 0.8
n <- 200
x <- t(replicate(n, rar1(p,psi)))
Z <- bandSparse(p, p, (-1):1,
                list(rep(1,p-1),
                     rep(1,p),
                     rep(1,p-1)))
```
The graphical structure of the data is contained in `Z`, which shows
the non-zero elements of the precision matrix. 
Such information is typically known in real-world problems.
```r
Z
# 5 x 5 sparse Matrix of class "dgCMatrix"
#               
# [1,] 1 1 . . .
# [2,] 1 1 1 . .
# [3,] . 1 1 1 .
# [4,] . . 1 1 1
# [5,] . . . 1 1
```
The exact dependence-structure is however typically unknown.
GraphSPME therefore estimates a non-parametric estimate of the precision matrix
using the `prec_sparse()` function.
```r
library(GraphSPME)
prec_est <- prec_sparse(x, Z)
prec_est
# 5 x 5 sparse Matrix of class "dgCMatrix"
#                                   
# [1,]  0.94 -0.85  .     .     .   
# [2,] -0.83  1.73 -0.78  .     .   
# [3,]  .    -0.86  1.58 -0.73  .   
# [4,]  .     .    -0.70  1.54 -0.78
# [5,]  .     .     .    -0.76  1.02
```
Note that GraphSPME allows working in very high dimensions:
```r
frobenius_norm <- function(M) sum(M^2)
set.seed(123)
p <- 10000
x <- t(replicate(n, rar1(p,psi)))
Z <- bandSparse(p, p, (-1):1,
                list(rep(1,p-1),
                     rep(1,p),
                     rep(1,p-1)))
dim(Z)
# [1] 10000 10000
start.time <- Sys.time()
prec_est <- prec_sparse(x, Z)
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
# Time difference of 0.9 secs
# We can compare with the population precision
prec_pop <- bandSparse(p, p, (-1):1, 
                       list(rep(-psi,p-1),
                            c(1, rep(1+psi^2,p-2), 1),
                            rep(-psi,p-1)))
frobenius_norm(prec_pop-prec_est)
# [1] 533
```

## Dependencies
GraphSPME is built on the linear algebra library [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
and utilizes [Rcpp](https://github.com/RcppCore/Rcpp) for bindings to R.
Bindings to Python are done via PyBind.
In particular the sparse matrix class in Eigen is extensively utilized to obtain efficient and scalable result.

- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) Linear algebra
- [Rcpp](https://github.com/RcppCore/Rcpp) for the R-package


## Main idea

GraphSPME combines the inversion scheme in Le (2021) with the regularized 
covariance estimate of Touloumis (2015) as described in <write paper>.
The package leverages Eigen to obtain scalable result by numerically taking advantage
of the graphical nature of the problem.

- Le (2021) introduces a sparse precision matrix estimate from the ml-estimate covariance matrix.
The method utilizes the knowledge of a graphical structure beneath the realized data.
- Touloumis (2015) finds asymptotic closed form results for schrinkage of the frequentist covariance estimate.
  

