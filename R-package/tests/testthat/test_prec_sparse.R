# Tests for prec_sparse()

library(GraphSPME)
library(Matrix)

p <- 10
n <- 100
X <- matrix(rnorm(p * n), nrow = n, ncol = p)
Z <- bandSparse(p, p, 0:0, list(rep(1, p)))
Z_wrong_sparsity <- as(Z, "dtCMatrix") # ensure wrong sparsity class
Z_correct_sparsity <- as(Z, "dgCMatrix") # ensure correct sparsity class

test_that("R throws an error for wrong sparsity type", {
  expect_error(
    prec_sparse(X, Z_wrong_sparsity),
    "Z must be a sparse matrix of type dgCMatrix"
  )
})

test_that("R throws an error for wrong data input type", {
  X_df <- data.frame(x)
  expect_error(
    prec_sparse(X_df, Z_correct_sparsity),
    "X must be a matrix"
  )
})

test_that("R throws an error when data and neighbourhood structure does not match", {
  X_extended <- cbind(X, rnorm(n))
  expect_error(prec_sparse(X_extended, Z_correct_sparsity),
    "ncol(X) must equal both dim(Z)",
    fixed = TRUE
  )
})
