library(GraphSPME)

test_that("cov_ml(X) is the same as R's cov(X)", {
  X = matrix(rnorm(36), nrow=6)
  expect_equal(cov_ml(X), cov(X))
})