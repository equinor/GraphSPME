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
  

