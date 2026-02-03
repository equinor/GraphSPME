# GraphSPME
[![Wheels](https://github.com/equinor/GraphSPME/actions/workflows/wheels.yml/badge.svg)](https://github.com/equinor/GraphSPME/actions/workflows/wheels.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> :warning: **The GraphSPME project was archived on 2026-02-03. It has been
> superseded by [Graphite Maps](https://github.com/equinor/graphite-maps), and
> this repository is now deprecated and no longer maintained.**

**High dimensional precision matrix estimation with a known graphical structure**

- [x] Works in very high dimensions
- [x] Non-parametric
- [x] Asymptotic regularization 🖖
- [x] Lightning fast
- [x] Both Python and R

Available as header-only in C++, as a Python-package, or as an R-package.


## Installation

**Python:**

GraphSPME is available on [PyPI](https://pypi.org/project/GraphSPME/):

```text
pip install GraphSPME
```

**R:**

Soon:
- [ ] R-package on CRAN

In the meantime, follow the [developer instructions](#developing) for installing GraphSPME. 
Note that GraphSPME relies heavily on [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page).
See [dependencies](#dependencies) for details.


## Example code and documentation

See [GraphSPME-examples](GraphSPME-examples) for R and Python example code.

We use the [AR1 process](https://en.wikipedia.org/wiki/Autoregressive_model) as an example, as it has known covariance, precision, and graphical structure.
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

GraphSPME is built on the linear algebra library [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). In particular the sparse matrix class in Eigen is extensively utilized to obtain efficient and scalable result.
For the R-package, [RcppEigen](https://github.com/RcppCore/Rcpp) is employed for both bindings to R and access to Eigen (no manual installation needed).
Bindings to [Python](https://pybind11.readthedocs.io/) are done via PyBind, and here Eigen must be installed manually beforehand.

- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) Linear algebra
- [Rcpp](https://github.com/RcppCore/Rcpp) for the R-package
- [Pybind](https://pybind11.readthedocs.io/) for the Python-package


## Developing

Clone the repository
```
git clone git@github.com:equinor/GraphSPME.git
```

To build the R-package, open the `R-package.Rproj` as a project in RStudio. Under `Build` hit `install and restart`.
To build it manually, or for CRAN, use the `Makefile` by running
```
make Rbuild # to bundle, or 
make Rinstall # to bundle and install, or
make Rcheck # to bundle and check if package ready for CRAN
```

To build the Python-package, first make sure that Eigen is installed and available in the include path. See the [Eigen getting started documentation](https://eigen.tuxfamily.org/dox/GettingStarted.html) for details.
Then GraphSPME is installable in editable mode by
```
pip install -e ./python-package
```


## Main idea

GraphSPME combines the inversion scheme in [Le (2021)](https://arxiv.org/abs/2107.06815) with the regularized 
covariance estimate of [Touloumis (2015)](https://arxiv.org/abs/1410.4726) as described in the paper "[GraphSPME: Markov Precision Matrix Estimation and Asymptotic Stein-Type Shrinkage](https://arxiv.org/abs/2205.07584)."
The package leverages Eigen to obtain scalable result by numerically taking advantage
of the graphical nature of the problem.

- The paper "[High-dimensional Precision Matrix Estimation with a Known Graphical Structure](https://arxiv.org/abs/2107.06815)" by Le et al. (2021) introduces a sparse precision matrix estimate from the ml-estimate covariance matrix.
The method utilizes the knowledge of a graphical structure beneath the realized data.
- The paper "[Nonparametric Stein-type Shrinkage Covariance Matrix Estimators in High-Dimensional Settings](https://arxiv.org/abs/1410.4726)" by Touloumis (2015) finds asymptotic closed form results for schrinkage of the frequentist covariance estimate.
  

