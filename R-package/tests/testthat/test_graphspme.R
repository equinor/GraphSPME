library(GraphSPME)
library(Matrix)

test_that("cov_ml(X) is the same as R's cov(X)", {
  X <- matrix(rnorm(36), nrow = 6)
  expect_equal(.cov_ml(X), cov(X))
})

test_that("create_bi satisfies Bi x wi1 = wi for simple band matrix", {
  p <- 5
  Z <- bandSparse(
    n = p, m = p, k = (-1):1,
    diagonals = list(rep(1, p - 1), rep(1, p), rep(1, p - 1))
  )
  B0 <- .create_bi(Z, 0)
  w01 <- c(4, 5)
  expected <- as(matrix(c(4, 5, 0, 0, 0)), "dgeMatrix")
  expect_equal(B0 %*% w01, expected)

  B1 <- .create_bi(Z, 1)
  w11 <- c(4, 5, 6)
  expected <- as(matrix(c(4, 5, 6, 0, 0)), "dgeMatrix")
  expect_equal(B1 %*% w11, expected)
})
