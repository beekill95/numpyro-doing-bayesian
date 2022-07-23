# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %cd ..
# %load_ext autoreload
# %autoreload 2

# +
from __future__ import annotations

import arviz as az
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro_glm.utils.dist as dist_utils
from numpyro.infer import MCMC, NUTS
import pandas as pd
import seaborn as sns

numpyro.set_host_device_count(4)
# -

# # Chapter 10: Model Comparison and Hierarchical Modeling
# ## Solution by MCMC
# ### Nonhierarchical MCMC Computation of each model's Marginal Likelihood


# #### Implementation

# +
def model(y: jnp.ndarray):
    nb_obs = y.shape[0]

    theta = numpyro.sample(
        'theta', dist_utils.beta_dist_from_omega_kappa(0.75, 12))

    with numpyro.plate('obs', nb_obs) as idx:
        numpyro.sample('y', dist.Bernoulli(theta), obs=y[idx])


kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.r_[jnp.zeros(3), jnp.ones(6)],
)
mcmc.print_summary()
# -

idata = az.from_numpyro(mcmc)
az.plot_trace(idata)

# Compute $p(D)$ by following equation 10.8.

# +
from scipy.stats import beta  # noqa

theta = idata['posterior']['theta'].values.flatten()
theta_mean = np.mean(theta)
theta_sd = np.std(theta)

# Convert mean, std to a and b of Beta distribution.
sd_squared = theta_sd**2
a_posterior = theta_mean * (theta_mean * (1 - theta_mean) / sd_squared - 1)
b_posterior = (1 - theta_mean) * (theta_mean *
                                  (1 - theta_mean) / sd_squared - 1)
one_over_pD = np.mean(beta.pdf(theta, a_posterior, b_posterior)
                      / (theta**6 * (1-theta)**3
                         * beta.pdf(theta, 0.75 * (12 - 2) + 1, (1 - 0.75) * (12 - 2) + 1)))
print(1. / one_over_pD)
