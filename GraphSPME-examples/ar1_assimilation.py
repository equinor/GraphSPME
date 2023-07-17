import functools
import graphspme
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from scipy import sparse

matplotlib.use("Qt5Agg")


# zero mean AR1
def rar1(n, psi):
    x = np.empty([n])
    x[0] = np.random.normal(0, 1 / math.sqrt(1 - psi**2), 1)
    for i in range(1, n):
        x[i] = psi * x[i - 1] + np.random.normal(0, 1, 1)
    return x


# Some random forward model
def G(mu_t0, psi):
    u = rar1(len(mu_t0), psi=psi)
    mu_t1 = mu_t0 + u
    return mu_t1


if __name__ == "__main__":
    np.random.seed(1)
    n = 200
    p = 100
    fig, axes = plt.subplots(1, 3)
    for psi, ax in zip([0.0, 0.2, 0.8], axes):
        # Sample the prior at t0
        mu_t0_t0_sample = np.tile(rar1(p, psi), (n, 1))

        # Bring forward to t1: Gives sample at t1
        g_bound = functools.partial(G, psi=psi)
        mu_t1_t0_sample = np.apply_along_axis(g_bound, 1, mu_t0_t0_sample)

        # Calculate prior 1|0 precision
        # Defining non-zero elements
        diagonals = [[1] * p, [1] * (p - 1), [1] * (p - 1)]
        Z = sparse.diags(diagonals, [0, -1, 1], format="csr")

        prec_t1_t0 = graphspme.prec_sparse(mu_t1_t0_sample, Z, True)

        mu_t1_t0 = mu_t1_t0_sample.mean(axis=0).reshape((p, 1))
        nu_t1_t0 = prec_t1_t0 @ mu_t1_t0

        # Calculate posterior 1|1 nu and precision
        # Using information filter equations
        # d is a sensor point at the middle p/2
        ind = math.ceil(p / 2)
        d_middle_t1 = np.array([3])
        sd_d = 1.3
        prec_d = np.array([1 / sd_d**2], ndmin=2)

        M = np.array([0] * p, ndmin=2)
        M[0, ind] = 1

        nu_t1_t1 = nu_t1_t0 + (np.transpose(M) @ prec_d @ d_middle_t1).reshape(
            (p, 1)
        )
        prec_t1_t1 = prec_t1_t0 + np.transpose(M) @ prec_d @ M
        mu_t1_t1 = np.linalg.inv(prec_t1_t1) * nu_t1_t1.reshape((p, 1))

        ax.plot(mu_t1_t1 - mu_t1_t0, "o", mfc="none")
        ax.set(
            title=f"True dependence, psi: {psi}",
            xlabel="state-element",
            ylabel="posterior - prior",
        )
    plt.show()
