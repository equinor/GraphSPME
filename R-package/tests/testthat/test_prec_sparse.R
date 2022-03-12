# Tests for prec_sparse()

library(GraphSPME)
library(Matrix)

p <- 10
n <- 100
X <- matrix(rnorm(p * n), nrow = n, ncol = p)
Graph <- bandSparse(p, p, 0:0, list(rep(1, p)))
Graph_wrong_sparsity <- as(Graph, "dtCMatrix") # ensure wrong sparsity class
Graph_correct_sparsity <- as(Graph, "dgCMatrix") # ensure correct sparsity class

test_that("R throws an error for wrong sparsity type", {
  expect_error(
    prec_sparse(X, Graph_wrong_sparsity),
    "Graph must be a sparse matrix of type dgCMatrix"
  )
})

test_that("R throws an error for wrong data input type", {
  X_df <- data.frame(X)
  expect_error(
    prec_sparse(X_df, Graph_correct_sparsity),
    "X must be a matrix"
  )
})

test_that("R throws an error when data and neighbourhood structure does not match", {
  X_extended <- cbind(X, rnorm(n))
  expect_error(prec_sparse(X_extended, Graph_correct_sparsity),
    "ncol(X) must equal both dim(Graph)",
    fixed = TRUE
  )
})

test_that("R throws an error for wrong input type in markov_order", {
  markov_order <- "string"
  expect_error(
    prec_sparse(X, Graph_correct_sparsity, markov_order),
    "markov_order must be a non-negative whole number"
  )
})

test_that("R throws an error for wrong numeric type in markov_order", {
  markov_order <- 1.2
  expect_error(
    prec_sparse(X, Graph_correct_sparsity, markov_order),
    "markov_order must be a non-negative whole number"
  )
})

test_that("R throws an error for negative int in markov_order", {
  markov_order <- -1
  expect_error(
    prec_sparse(X, Graph_correct_sparsity, markov_order),
    "markov_order must be a non-negative whole number"
  )
})
