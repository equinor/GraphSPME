# precision estimation
# 1. graphspme
# 2. circle theorem adjustment
# 3. chol elements estimation -- always inside SPD family
# 4. precision directly

library(GraphSPME) # load from namespace to avoid conflicts with cpp external
library(Matrix)


frobenius_norm <- function(M) sum(M^2)

# Zero mean AR1
rar1 <- function(n, psi){
    x <- numeric(n)
    x[1] <- rnorm(1, 0, 1/sqrt(1-psi^2))
    w <- rnorm(n-1)
    for(i in 2:n){
        x[i] <- psi * x[i-1] + w[i-1]
    }
    return(x)
}

set.seed(123)
p <- 200
psi <- 0.9
n <- 200
x <- t(replicate(n, rar1(p,psi)))

prec_pop <- bandSparse(p, p, (-1):1, 
                       list(rep(-psi,p-1),
                            c(1, rep(1+psi^2,p-2), 1),
                            rep(-psi,p-1)))

# 1. graphspme, likelihood free initial
Z <- bandSparse(p, p, (-1):1,
                list(rep(1,p-1),
                     rep(1,p),
                     rep(1,p-1)))
prec_1 <- GraphSPME::prec_sparse(
    x,
    Z,
    markov_order = 1,
    shrinkage = TRUE,
    symmetrization = TRUE
)


# 2. Gershgorin circle theorem
prec_2 <- prec_1
eps <- 1e-2
diagonal_adjustment <- numeric(p)
offdiag_abs_sum <- Matrix::rowSums(abs(prec_2)) - abs(diag(prec_2))
for(i in 1:p){
    if(offdiag_abs_sum[i] > prec_2[i,i]){
        prec_2[i,i] <- offdiag_abs_sum[i] + eps
    }
}


# 3. Optimize under SPD and with optimized in-fill
# - find optimized permutation
perm_indices <- GraphSPME:::.compute_amd_ordering(prec_2)
Perm = t(sparseMatrix(seq_along(perm_indices), perm_indices+1, x=1))
L <- GraphSPME:::.cholesky_factor(prec_2, perm_indices)
choleskyLowerGraph <- L

# - 4. optimize under spd
paramIntoCholLower <- function(theta, GraphL){
    graphSumamry <- summary(GraphL)
    L <- sparseMatrix(
        i = graphSumamry$i,
        j = graphSumamry$j,
        x = theta,
        dims = dim(GraphL)
    )
    return(as(L, "dgCMatrix"))
}

funL <- function(thetaL, x, GraphL, perm_indices){
    L <- paramIntoCholLower(thetaL, GraphL)
    GraphSPME:::.dmrfL(x, L, perm_indices)
}

gradL <- function(thetaL, x, GraphL, perm_indices){
    L <- paramIntoCholLower(thetaL, GraphL)
    GraphSPME:::.dmrfL_grad(x, L, L, perm_indices)
}

hessL <- function(thetaL, x, GraphL, perm_indices){
    L <- paramIntoCholLower(thetaL, GraphL)
    GraphSPME:::.dmrfL_hess(x, L, L, perm_indices)
}

prec_chol_L_opt <- function(x, GraphL, L_estimate_prev, perm_indices)
{
    # start from previous estimate
    thetaL <- summary(L_estimate_prev)$x
    opt <- nlminb(
        thetaL,
        funL,
        gradL,
        hessL,
        control=list(trace=0, eval.max=1e3, iter.max=1e3),
        x=x,
        GraphL = GraphL,
        perm_indices = perm_indices
    )
    print(opt$message)
    L <- paramIntoCholLower(opt$par, GraphL)
    return(L)
}

L_opt <- prec_chol_L_opt(x, choleskyLowerGraph, L, perm_indices)

# - 5. optimize with no in-fill
paramIntoPrec <- function(theta, Graph){
    triplets <- summary(Graph)
    triplets_lower_triagonal <- subset(triplets, i>=j)
    Prec_lower_tri <- sparseMatrix(
        i = triplets_lower_triagonal$i,
        j = triplets_lower_triagonal$j,
        x = theta,
        dims=dim(Graph)
    )
    Prec_upper_tri <- t(Prec_lower_tri)
    diag(Prec_upper_tri) = 0
    Prec = Prec_lower_tri + Prec_upper_tri
    return(as(Prec, "dgCMatrix"))
}

# could pass permutation indices to avoid optimizing amd every time
fun <- function(theta, x, Graph, perm_indices){
    Prec = paramIntoPrec(theta, Graph)
    GraphSPME:::.dmrf(x, Prec, perm_indices)
}

grad <- function(theta, x, Graph, perm_indices){
    Prec = paramIntoPrec(theta, Graph)
    GraphSPME:::.dmrf_grad(x, Prec, Prec)
}

hess <- function(theta, x, Graph, perm_indices){
    Prec = paramIntoPrec(theta, Graph)
    GraphSPME:::.dmrf_hess(Prec, Prec)
}

prec_opt <- function(x, Graph, Prec_estimate_prev, perm_indices)
{
    # start from previous estimate
    theta <- subset(summary(Prec_estimate_prev), i >= j)$x
    opt <- nlminb(
        theta,
        fun,
        grad,
        hess,
        control=list(trace=0, eval.max=1e3, iter.max=1e3),
        x=x,
        Graph = Graph,
        perm_indices=perm_indices
    )
    print(opt$message)
    Prec <- paramIntoPrec(opt$par, Graph)
    return(Prec)
}

prec_4 <- t(Perm) %*% (L_opt %*% t(L_opt)) %*% Perm
prec_5 <- prec_opt(x, Z, prec_4, perm_indices)
# if no in-fill, then Prec_5 should equal Prec_4

# check for different orders of n, that prec_5 converges
frobenius_norm(prec_pop-prec_5)
GraphSPME:::.dmrf(x, prec_pop, perm_indices)
GraphSPME:::.dmrf(x, prec_1, perm_indices)
GraphSPME:::.dmrf(x, prec_2, perm_indices)
GraphSPME:::.dmrf(x, prec_4, perm_indices)
GraphSPME:::.dmrf(x, prec_5, perm_indices)

# check updates visually through information filter
#' Information filter data assimilation
update_parameters <- function(yi, H, mu, Prec, Prec_y){
    eta <- Prec %*% mu
    eta <- eta + t(H) %*% Prec_y %*% yi
    Prec <- Prec + t(H) %*% Prec_y %*% H
    return(list(
        "eta" = eta,
        "Prec" = Prec
    ))
}

#' Computationally efficient transform to mean from eta
retrieve_mean <- function(eta, Prec){
    mu <- Matrix::solve(Prec, eta)[,1]
    return(mu)
}

# y is a sensor point at the middle p/2
ind <- ceiling(p/2)
y <- 3
sd_y <- 1.3
Prec_y <- matrix(1/sd_y^2, nrow=1, ncol=1)
# direct observation matrix of ind
H <- matrix(0, nrow=1, ncol=p)
H[1,ind] <- 1

# prec_u first
mu_t1_t0 <- colMeans(x)
#Prec <- par_est$Prec
# Assimilate data, updating canonical parametrisation
update <- update_parameters(y, H, mu_t1_t0, prec_pop, Prec_y)
eta <- update$eta
Prec_u <- update$Prec
mu_t1_t1 <- retrieve_mean(eta, Prec_u)

x_u_1 <- x_u_2 <- x
prec_5_u <- prec_5 + t(H) %*% Prec_y %*% H
for(i in 1:n){
    eps <- rnorm(1,sd=sd_y)
    x_i_nu1 <- prec_5 %*% x[i,]
    x_i_u1 <- x_i_nu1 + t(H) %*% Prec_y %*% (y+eps)
    x_u_1[i,] <- Matrix::solve(prec_5_u, x_i_u1)[,1]
    
    x_i_nu2 <- prec_pop %*% x[i,]
    x_i_u2 <- x_i_nu2 + t(H) %*% Prec_y %*% (y+eps)
    x_u_2[i,] <- Matrix::solve(Prec_u, x_i_u2)[,1]
}

par(mfrow=c(2,3))
matplot(t(x), type="l", main="ensemble prior")
matplot(t(x_u_1), type="l", main="ensemble posterior, using estimated precision")
plot(colMeans(x_u_1) - colMeans(x), main="average update, using estimated precision")
matplot(t(x), type="l", main="ensemble prior")
matplot(t(x_u_2), type="l", main="ensemble posterior, using population precision")
plot(colMeans(x_u_2) - colMeans(x), main="average update, using population precision")
par(mfrow=c(1,1))
