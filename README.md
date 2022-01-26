# Graphical Sparse Precision Matrix Estimation

**GraphSPME** employs the approach of two methods to create efficient and
computationally fast estimates of precision matrices.

**High dimensional precision matrix estimation with a known graphical structure**

Le 2021 introduces an estimate using the ml-estimate covariance matrix.
The method utilizes the knowledge of a graphical structure beneath the realized data.

**Improved high dimensional covariance estimation**

Touloumis 2014 finds asymptotic results for schrinkage to ml-estimate of covariance.
The resulting matrix estimates are 

1. non-singular
2. well-conditioned
3. invariant to permutations of the order of the p variables
4. consistent to departures from a multivariate normal model
5. not necessarily sparse
6. expressed in closed form
7. computationally cheap regardless of `p`

**GraphSPME** employs the inversion scheme in Le 2021 using the regularized 
covariance estimate of Touloumis 2014.
