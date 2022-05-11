# Experiments from section 5 in GraphSPME paper
# Using GraphSPME
# Illustrated with a mixed-effect ar-p model, with p unknown
# p is estimated using built-in aic
# random and fixed effects are retrieved

library(GraphSPME)
library(Matrix)
library(dplyr)
library(ggplot2)

options(digits=3)

rarp <- function(n, p, u, psi){
    # random effect ar-p
    x <- numeric(n+p)
    x[1:p] <- rnorm(p)
    for(t in (p+1):(n+p)){
        x[t] <- u[t-p] * sum(psi*x[(t-p):(t-1)]) + rnorm(1)
    }
    return(x[(p+1):(n+p)])
}

set.seed(123)
n <- 100
ntime <- 100
p_true <- 3
u <- runif(ntime, 0, 1)
psi <- 1:p_true/sum(1:p_true)
xtr <- t(replicate(n, rarp(ntime, p_true, u, psi)))
max_order <- 8 # p - 1
pval <- ltr <- lte <- numeric(max_order+1)
G <- bandSparse(ntime, ntime, 
                (-1):1, 
                sapply((-1):1, function(j) rep(1,ntime-abs(j)))
)
G[1:5,1:5]

nll <- aic <- numeric(max_order+1)
for(p in 0:max_order){
    Prec_est <- prec_sparse(xtr, G, markov_order=p)
    nll[p+1] <- prec_nll(xtr, Prec_est)
    aic[p+1] <- prec_aic(xtr, Prec_est)
}
df <- data.frame(
    "order" = c(0:max_order, 0:max_order),
    "loss" = c(nll, aic),
    "type" = c(rep("train",max_order+1), rep("aic",max_order+1))
)
p_markov_estimation <- 
    df %>% 
    ggplot(aes(x=order, y=loss, group=type)) + 
    geom_point(aes(colour=type)) + 
    geom_line(aes(colour=type)) + 
    ggtitle(
        "AIC adjusted quasi-MLE objective for precision estimate",
        subtitle = paste0("True Markov order: ", p_true)
    ) + 
    xlab("Markov order") + 
    theme_bw()
p_markov_estimation
if(F){
    pdf("markov_order_estimation.pdf", width=6, height=3.5)
    p_markov_estimation
    dev.off()
}
markov_order <- which.min(aic)-1

# create final estimate
p_est <- which.min(aic)-1
Prec <- prec_sparse(xtr, G, 3)
Prec[1:5,1:5]


## ---- retrieve fixed and random effects ----
mean_cond <- function(i,x, Prec){
    # mu - q/Qii sum_ne(i) Prec_ij(xj-mu_j)
    if(i==1){
        return(0)
    }else{
        -1/Prec[i,i] * (Prec[i,1:(i-1)] %*% x[1:(i-1)])
    }
}

mse_psi <- function(psi, y, x){
    mean((y-x%*%psi)^2)
}

mse_u <- function(u, y, preds){
    sum((y-t(t(preds)*u))^2) / length(preds)
}

dmat <- c()
for(i in 1:p_est){
    cind <- (1+i-1):(ntime-p_est+i-1)
    dmat <- cbind(
        dmat,
        as.vector(t(xtr[,cind]))
    )
}
dmat <- as.matrix(dmat)
mu_prec <- c()
for(i in 1:n){
    mu <- sapply(1:ntime, function(j) mean_cond(j, x=xtr[i,], Prec=Prec))
    mu_prec <- c(mu_prec, mu[-(1:p_est)]) 
}

psi_est <- rep(1,p_est)/p_est
u_est <- rep(0.5, ntime-p_est)
# estimate u
preds <- dmat %*% psi_est
preds <- matrix(preds, ncol=ntime-p_est, byrow=T)
optu <- nlminb(u_est, mse_u, y=matrix(mu_prec, ncol=ntime-p_est, byrow=T), preds=preds)
u_est <- optu$par
# estimate psi
psi_target <- mu_prec / rep(u_est,n)
opt_psi <- nlminb(psi_est, mse_psi, y=psi_target, x=dmat)
psi_est <- opt_psi$par

df_u <- data.frame(
    "time" = rep((p_est+1):ntime,2),
    "process" = c(u[(p_est+1):ntime], u_est),
    "type" = c(rep("true", ntime-p_est), rep("estimate", ntime-p_est))
)
p_u <- 
    df_u %>% 
    ggplot(aes(x=time, y=process, group=type)) + 
    geom_point(aes(colour=type)) + 
    geom_line(aes(colour=type)) + 
    ggtitle("Random effects") + 
    theme_bw()

df_psi <- data.frame(
    fixed_effect = rep(as.factor(paste0("psi",1:p_est)),2),
    value = c(rev(psi), rev(psi_est)),
    type = c(rep("exact", p_est), rep("estiamte", p_est))
)
p_psi <- 
    df_psi %>%
    ggplot(aes(x=fixed_effect, y=value, group=type)) + 
    geom_point(aes(colour=type, shape=type)) + 
    ggtitle("Fixed effects") + 
    theme_bw()

gridExtra::grid.arrange(
    p_u, p_psi,
    ncol=2
)
