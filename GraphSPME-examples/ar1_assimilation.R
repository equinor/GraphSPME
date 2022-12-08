renv::load("~/repos/GraphSPME")
library(GraphSPME)
library(Matrix)

# Do a super-simple 1-dim example first!
# zero mean AR1
rar1 <- function(n, psi){
    x <- numeric(n)
    x[1] <- rnorm(1, 0, 1/sqrt(1-psi^2))
    for(i in 2:n){
        x[i] <- psi*x[i-1] + rnorm(1)
    }
    return(x)
}

# Some arbitrary forward model
G <- function(mu_t0, psi){
    u <- rar1(length(mu_t0), psi)
    mu_t1 <- mu_t0 + u
    return(mu_t1)
}

set.seed(123)
psi_values <- c(0,0.2,0.8)
par(mfrow=c(1,3))
for(psi in psi_values){
    p <- 100
    n <- 200
    # Sample the prior at t0
    mu_t0_t0_sample <- t(replicate(n, rar1(p,psi)))
    # Bring forward to t1: Gives sample at t1
    mu_t1_t0_sample <- t(sapply(1:n, function(i) G(mu_t0_t0_sample[i,], psi=psi)))

    # Calculate prior 1|0 precision
    # Defining non-zero elements
    Z <- bandSparse(p, p,
                    (-1):1,
                    list(rep(1,p-1),
                         rep(1,p),
                         rep(1,p-1))
    )
    mu_t1_t0 <- colMeans(mu_t1_t0_sample)
    prec_t1_t0 <- prec_sparse(mu_t1_t0_sample, Graph=Z)
    prec_t1_t0
    nu_t1_t0 <- prec_t1_t0 %*% mu_t1_t0

    # Calculate posterior 1|1 nu and precision
    # Using information filter equations
    # d is a sensor point at the middle p/2
    ind <- ceiling(p/2)
    d_middle_t1 <- 3
    sd_d <- 1.3
    prec_d <- matrix(1/sd_d^2, nrow=1, ncol=1)

    # direct observation matrix of ind
    M <- matrix(0, nrow=1, ncol=p)
    M[1,ind] <- 1

    nu_t1_t1 <- nu_t1_t0 + t(M) %*% prec_d %*% d_middle_t1
    prec_t1_t1 <- prec_t1_t0 + t(M) %*% prec_d %*% M
    mu_t1_t1 <- solve(prec_t1_t1) %*% nu_t1_t1

    # Compare posterior estimate with prior
    plot(mu_t1_t1 - mu_t1_t0,
         xlab="state-element",
         ylab="posterior - prior",
         main = paste0("True dependence, psi: ", psi)
    )
}
