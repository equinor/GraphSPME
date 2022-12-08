# Tests for GraphSPME:::.ldl_fbmod()
library(GraphSPME)
library(Matrix)

p <- 3
M_random <- matrix(runif(p^2), p, p)
# ensure symmetry
M_random <- 0.5*(M_random + t(M_random))
ev <- eigen(M_random)
lambda <- ev$values
V <- ev$vectors
# ensure first eigenvalue negative, rest positive
M_snd <- V %*% diag(c(-1,pmax(lambda[-1],1))) %*% t(V)
M_snd <- as(M_snd, "dgCMatrix")
# ensure all eigenvalues positive
M_spd <- V %*% diag(pmax(lambda,1)) %*% t(V)
M_spd <- as(M_spd, "dgCMatrix")

test_that("ldl_fbmod modifies snd matrix and returns spd matrix", {
    M_mod <- GraphSPME:::.ldl_fbmod(M_snd)
    expect_true(all(eigen(M_mod)$values >= 0))
})

test_that("ldl_fbmod returns the same matrix when found to be spd>eps", {
    M_mod <- GraphSPME:::.ldl_fbmod(M_spd, eps=1e-3)
    expect_equal(M_mod, M_spd)
})
