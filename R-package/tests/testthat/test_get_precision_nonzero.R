# Tests for GraphSPME:::.get_precision_nonzero()
library(GraphSPME)
library(Matrix)

p <- 10
Neighbours <- bandSparse(
  p, p,
  (-1):1,
  list(
    rep(1, p - 1),
    rep(1, p),
    rep(1, p - 1)
  )
)

Ip <- as(diag(p), "dgCMatrix")

test_that("get_precision_nonzero returns identity at markov order 0", {
  expect_equal(GraphSPME:::.get_precision_nonzero(Neighbours, 0), Ip)
})

test_that("get_precision_nonzero returns Neighbours at markov order 1", {
  expect_equal(GraphSPME:::.get_precision_nonzero(Neighbours, 1), Neighbours)
})
