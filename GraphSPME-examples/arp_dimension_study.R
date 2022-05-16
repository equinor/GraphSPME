# Experiments from section 6 in GraphSPME paper
# AR-p model with varying sample-size, time (and thus dimension), and auto-regressive
# parameter p

library(ts.extend)
library(GraphSPME)
library(Matrix)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(EigenR)

options(digits=3)

fbnorm <- function(M) sqrt(sum(M^2))

quantileMat <- function(M, q){
    p <- ncol(M)
    n <- nrow(M)
    sapply(1:p, function(j){quantile(M[,j], probs=q)})
}

set.seed(123)
B <- 100

## INCREASING DIMENSION
#Set the parameters
n <- 100
nd <- 50
ntimes <- round(exp(seq(log(10),log(300), length.out=nd)))
p <- 1
AR    <- rep(0.8,p)/p 

fb_cov_s <- fb_cov_shrink <- fb_cov_le <- fb_cov_gspme <- 
    fb_prec_s <- fb_prec_shrink <- fb_prec_le <- fb_prec_gspme <- matrix(nrow=B,ncol=nd)
pb <- txtProgressBar(0, nd*B)
for(b in 1:B){
    for(i in 1:nd){
        # Find population quantities
        ntime <- ntimes[i]
        Sigma <- ARMA.var(ntime, ar=AR)
        Prec <- solve(Sigma)
        G <- bandSparse(ntime, ntime, 
                        (-p):p, 
                        sapply((-p):p, function(order) rep(1,ntime-abs(order)))
        )
        # Sample data
        xtr <- ts.extend::rGARMA(n,ntime, ar=AR)
        # Compute different estimates
        S <- cov(xtr)
        S_shrink <- cov_shrink_spd(xtr)
        Prec_le <- prec_sparse(xtr, G, shrinkage=F, symmetrization = F)
        Prec_gspme <- prec_sparse(xtr, G)
        # Evaluate loss for covariance and precision
        fb_cov_s[b,i] <- fbnorm(Sigma - S)
        fb_cov_shrink[b,i] <- fbnorm(Sigma - S_shrink)
        fb_cov_le[b,i] <- fbnorm(Sigma - solve(Prec_le))
        fb_cov_gspme[b,i] <- fbnorm(Sigma - solve(Prec_gspme))
        fb_prec_s[b,i] <- fbnorm(Prec - Eigen_pinverse(S))
        fb_prec_shrink[b,i] <- fbnorm(Prec - solve(S_shrink))
        fb_prec_le[b,i] <- fbnorm(Prec - Prec_le)
        fb_prec_gspme[b,i] <- fbnorm(Prec - Prec_gspme)
        setTxtProgressBar(pb, value=(nd-1)*b + i)
    }   
}
close(pb)

df_cov <- data.frame(
    "dimension" = rep(ntimes, 4),
    "frobenius" = c(sapply(list(fb_cov_s, fb_cov_shrink, fb_cov_le, fb_cov_gspme), colMeans)),
    "low" = c(sapply(list(fb_cov_s, fb_cov_shrink, fb_cov_le, fb_cov_gspme), quantileMat, q=0.05)),
    "high" = c(sapply(list(fb_cov_s, fb_cov_shrink, fb_cov_le, fb_cov_gspme), quantileMat, q=0.95)),
    "Estimate" = c(rep("sample",nd), rep("shrinkage",nd), rep("Le",nd), rep("gspme",nd))
)
df_prec <- data.frame(
    "dimension" = rep(ntimes, 4),
    "frobenius" = c(sapply(list(fb_prec_s, fb_prec_shrink, fb_prec_le, fb_prec_gspme), colMeans)),
    "low" = c(sapply(list(fb_prec_s, fb_prec_shrink, fb_prec_le, fb_prec_gspme), quantileMat, q=0.05)),
    "high" = c(sapply(list(fb_prec_s, fb_prec_shrink, fb_prec_le, fb_prec_gspme), quantileMat, q=0.95)),
    "Estimate" = c(rep("SCV",nd), rep("shrinkage",nd), rep("Le",nd), rep("gspme",nd))
)
p_fb_cov <- 
    df_cov %>% 
    ggplot(aes(x=dimension, y=frobenius, group=Estimate)) + 
    geom_line(aes(colour=Estimate, linetype=Estimate), size=1) + 
    geom_ribbon(aes(ymin=low, ymax=high, fill=Estimate), alpha=0.2) + 
    ggtitle(
        "Covariance matrix"
    ) + 
    ylab("Frobenius norm") + 
    xlab("Dimension (AR-time)") + 
    scale_y_continuous(trans='log10') + 
    theme_bw()
p_fb_prec <- 
    df_prec %>% 
    ggplot(aes(x=dimension, y=frobenius, group=Estimate)) + 
    geom_line(aes(colour=Estimate, linetype=Estimate), size=1) +
    geom_ribbon(aes(ymin=low, ymax=high, fill=Estimate), alpha=0.2) + 
    ggtitle(
        "Precision matrix"
    ) + 
    ylab("Frobenius norm") + 
    xlab("Dimension (AR-time)") + 
    scale_y_continuous(trans='log10') + 
    theme_bw()
pcomb <- 
    ggarrange(
        p_fb_cov,
        p_fb_prec,
        ncol=2,
        common.legend = TRUE,
        legend="bottom"
    ) 
plot(pcomb)
if(F){
    ggexport(pcomb, filename = "fbnorm_vs_dim.pdf", width=7, height=3.5)
}


## INCREASING SAMPLE SIZE
#Set the parameters
nd <- 50
ns <- round(exp(seq(log(30),log(10000), length.out=nd)))
ntime <- 100
#ntimes <- round(exp(seq(log(10),log(1000), length.out=nd)))
p <- 1
AR    <- rep(0.8,p)/p 

fb_cov_s <- fb_cov_shrink <- fb_cov_le <- fb_cov_gspme <- 
    fb_prec_s <- fb_prec_shrink <- fb_prec_le <- fb_prec_gspme <- matrix(nrow=B, ncol=nd)
pb <- txtProgressBar(0, B*nd)
for(b in 1:B){
    for(i in 1:nd){
        # Find population quantities
        n <- ns[i]
        Sigma <- ARMA.var(ntime, ar=AR)
        Prec <- solve(Sigma)
        G <- bandSparse(ntime, ntime, 
                        (-p):p, 
                        sapply((-p):p, function(order) rep(1,ntime-abs(order)))
        )
        # Sample data
        xtr <- ts.extend::rGARMA(n,ntime, ar=AR)
        # Compute different estimates
        S <- cov(xtr)
        S_shrink <- cov_shrink_spd(xtr)
        Prec_le <- prec_sparse(xtr, G, shrinkage=F, symmetrization = F)
        Prec_gspme <- prec_sparse(xtr, G)
        # Evaluate loss for covariance and precision
        fb_cov_s[b,i] <- fbnorm(Sigma - S)
        fb_cov_shrink[b,i] <- fbnorm(Sigma - S_shrink)
        fb_cov_le[b,i] <- fbnorm(Sigma - solve(Prec_le))
        fb_cov_gspme[b,i] <- fbnorm(Sigma - solve(Prec_gspme))
        fb_prec_s[b,i] <- fbnorm(Prec - Eigen_pinverse(S))
        fb_prec_shrink[b,i] <- fbnorm(Prec - solve(S_shrink))
        fb_prec_le[b,i] <- fbnorm(Prec - Prec_le)
        fb_prec_gspme[b,i] <- fbnorm(Prec - Prec_gspme)
        setTxtProgressBar(pb, value=(nd-1)*b + i)
    }   
}
close(pb)

df_cov <- data.frame(
    "sample_size" = rep(ns, 4),
    "frobenius" = c(sapply(list(fb_cov_s, fb_cov_shrink, fb_cov_le, fb_cov_gspme), colMeans)),
    "low" = c(sapply(list(fb_cov_s, fb_cov_shrink, fb_cov_le, fb_cov_gspme), quantileMat, q=0.05)),
    "high" = c(sapply(list(fb_cov_s, fb_cov_shrink, fb_cov_le, fb_cov_gspme), quantileMat, q=0.95)),
    "Estimate" = c(rep("sample",nd), rep("shrinkage",nd), rep("Le",nd), rep("gspme",nd))
)
df_prec <- data.frame(
    "sample_size" = rep(ns, 4),
    "frobenius" = c(sapply(list(fb_prec_s, fb_prec_shrink, fb_prec_le, fb_prec_gspme), colMeans)),
    "low" = c(sapply(list(fb_prec_s, fb_prec_shrink, fb_prec_le, fb_prec_gspme), quantileMat, q=0.05)),
    "high" = c(sapply(list(fb_prec_s, fb_prec_shrink, fb_prec_le, fb_prec_gspme), quantileMat, q=0.95)),
    "Estimate" = c(rep("SCV",nd), rep("shrinkage",nd), rep("Le",nd), rep("gspme",nd))
)
names(df_cov)[1] <- names(df_prec)[1] <- "sample-size"
p_fb_cov <- 
    df_cov %>% 
    ggplot(aes(x=`sample-size`, y=frobenius, group=Estimate)) + 
    geom_line(aes(colour=Estimate, linetype=Estimate), size=1) + 
    geom_ribbon(aes(ymin=low, ymax=high, fill=Estimate), alpha=0.2) + 
    ggtitle(
        "Covariance matrix"
    ) + 
    ylab("Frobenius norm") + 
    xlab("Sample size") + 
    scale_y_continuous(trans='log10') + 
    scale_x_continuous(trans='log10') + 
    theme_bw()
p_fb_prec <- 
    df_prec %>% 
    ggplot(aes(x=`sample-size`, y=frobenius, group=Estimate)) + 
    geom_line(aes(colour=Estimate, linetype=Estimate), size=1) + 
    geom_ribbon(aes(ymin=low, ymax=high, fill=Estimate), alpha=0.2) + 
    ggtitle(
        "Precision matrix"
    ) + 
    ylab("Frobenius norm") + 
    xlab("Sample size") + 
    scale_y_continuous(trans='log10') +
    scale_x_continuous(trans='log10') + 
    theme_bw()
pcomb <- 
    ggarrange(
        p_fb_cov,
        p_fb_prec,
        ncol=2,
        common.legend = TRUE,
        legend="bottom"
    ) 
plot(pcomb)
if(F){
    ggexport(pcomb, filename = "fbnorm_vs_samplesize.pdf", width=7, height=3.5)
}


## INCREASING AUTO-REGRESSIVE PARAMETER
#Set the parameters
n <- 100
ntime <- 40
ps <- seq(0, ntime-1)
nd <- length(ps)

fb_cov_s <- fb_cov_shrink <- fb_cov_le <- fb_cov_gspme <- 
    fb_prec_s <- fb_prec_shrink <- fb_prec_le <- fb_prec_gspme <- 
    matrix(nrow=B,ncol=nd)
pb <- txtProgressBar(0, B*nd)
for(b in 1:B){
    for(i in 1:nd){
        # Find population quantities
        p <- ps[i]
        if(p>0){
            AR    <- rep(0.8,p)/p    
        }else{
            AR <- numeric(0)
        }
        Sigma <- ARMA.var(ntime, ar=AR)
        Prec <- solve(Sigma)
        G <- bandSparse(ntime, ntime, 
                        (-p):p, 
                        sapply((-p):p, function(order) rep(1,ntime-abs(order)))
        )
        G <- as(G, "dgCMatrix")
        # Sample data
        xtr <- ts.extend::rGARMA(n,ntime, ar=AR)
        # Compute different estimates
        S <- cov(xtr)
        S_shrink <- cov_shrink_spd(xtr)
        Prec_le <- prec_sparse(xtr, G, shrinkage=F, symmetrization=F)
        Prec_gspme <- prec_sparse(xtr, G)
        # Evaluate loss for covariance and precision
        fb_cov_s[b,i] <- fbnorm(Sigma - S)
        fb_cov_shrink[b,i] <- fbnorm(Sigma - S_shrink)
        fb_cov_le[b,i] <- fbnorm(Sigma - solve(Prec_le))
        fb_cov_gspme[b,i] <- fbnorm(Sigma - solve(Prec_gspme))
        fb_prec_s[b,i] <- fbnorm(Prec - Eigen_pinverse(S))
        fb_prec_shrink[b,i] <- fbnorm(Prec - solve(S_shrink))
        fb_prec_le[b,i] <- fbnorm(Prec - Prec_le)
        fb_prec_gspme[b,i] <- fbnorm(Prec - Prec_gspme)
        setTxtProgressBar(pb, value=(nd-1)*b+i)
    }   
}
close(pb)
df_cov <- data.frame(
    "sample_size" = rep(ps, 4),
    "frobenius" = c(sapply(list(fb_cov_s, fb_cov_shrink, fb_cov_le, fb_cov_gspme), colMeans)),
    "low" = c(sapply(list(fb_cov_s, fb_cov_shrink, fb_cov_le, fb_cov_gspme), quantileMat, q=0.05)),
    "high" = c(sapply(list(fb_cov_s, fb_cov_shrink, fb_cov_le, fb_cov_gspme), quantileMat, q=0.95)),
    "Estimate" = c(rep("sample",nd), rep("shrinkage",nd), rep("Le",nd), rep("gspme",nd))
)
df_prec <- data.frame(
    "sample_size" = rep(ps, 4),
    "frobenius" = c(sapply(list(fb_prec_s, fb_prec_shrink, fb_prec_le, fb_prec_gspme), colMeans)),
    "low" = c(sapply(list(fb_prec_s, fb_prec_shrink, fb_prec_le, fb_prec_gspme), quantileMat, q=0.05)),
    "high" = c(sapply(list(fb_prec_s, fb_prec_shrink, fb_prec_le, fb_prec_gspme), quantileMat, q=0.95)),
    "Estimate" = c(rep("SCV",nd), rep("shrinkage",nd), rep("Le",nd), rep("gspme",nd))
)
names(df_cov)[1] <- names(df_prec)[1] <- "auto-regressive parameter"
p_fb_cov <- 
    df_cov %>% 
    ggplot(aes(x=`auto-regressive parameter`, y=frobenius, group=Estimate)) + 
    geom_line(aes(colour=Estimate, linetype=Estimate), size=1) + 
    geom_ribbon(aes(ymin=low, ymax=high, fill=Estimate), alpha=0.2) + 
    ggtitle(
        "Covariance matrix"
    ) + 
    ylab("Frobenius norm") + 
    xlab("auto-regressive order") + 
    scale_y_continuous(trans='log10') + 
    theme_bw()
p_fb_prec <- 
    df_prec %>% 
    ggplot(aes(x=`auto-regressive parameter`, y=frobenius, group=Estimate)) + 
    geom_line(aes(colour=Estimate, linetype=Estimate), size=1) + 
    geom_ribbon(aes(ymin=low, ymax=high, fill=Estimate), alpha=0.2) + 
    ggtitle(
        "Precision matrix"
    ) + 
    ylab("Frobenius norm") + 
    xlab("auto-regressive order") + 
    scale_y_continuous(trans='log10') +
    theme_bw()
pcomb <- 
    ggarrange(
        p_fb_cov,
        p_fb_prec,
        ncol=2,
        common.legend = TRUE,
        legend="bottom"
    ) 
plot(pcomb)
if(F){
    ggexport(pcomb, filename = "fbnorm_vs_ar_order.pdf", width=7, height=3.5)
}

