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
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
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


# -

# ### Hierarchical MCMC Computation of Relative Model Probability

# +
def model_hier(y: jnp.ndarray):
    nb_obs = y.shape[0]

    model_prior = jnp.array([0.5, 0.5])
    m = numpyro.sample('m', dist.Categorical(model_prior))

    omega = jnp.array([0.25, 0.75])
    kappa = 12

    theta = numpyro.sample(
        'theta', dist_utils.beta_dist_from_omega_kappa(omega[m], kappa))

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        numpyro.sample('y', dist.Bernoulli(theta), obs=y[idx])


kernel = DiscreteHMCGibbs(NUTS(model_hier))
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.r_[jnp.zeros(3), jnp.ones(6)],
)
mcmc.print_summary()
# -

idata = az.from_numpyro(mcmc)
az.plot_trace(idata)
plt.tight_layout()

# +
fig: plt.Figure = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(nrows=2, ncols=2)

# Plot model index.
m = idata['posterior']['m'].values.flatten()
ax = fig.add_subplot(gs[:, 0])
az.plot_posterior(m, hdi_prob=.95, point_estimate='mean', ax=ax)
ax.set_title('Model Index')
ax.set_xlabel('m')

# Plot theta's posterior of the first model (m = 0).
# Index 0 corresponds to index 1 in R.
theta = idata['posterior']['theta'].values.flatten()
theta_m_0 = theta[m == 0]
p_m_0 = np.sum(m == 0) / m.size
ax = fig.add_subplot(gs[0, 1])
az.plot_posterior(theta_m_0, hdi_prob=.95, point_estimate='mode', ax=ax)
ax.set_title(f'$\\theta_{{m = 0}}. p(m = 0| D) = {p_m_0:.3f}$')
ax.set_xlabel('$\\theta$')

# Plot theta's posterior of the second model (m = 1).
# Index 1 corresponds to index 2 in R.
theta = idata['posterior']['theta'].values.flatten()
theta_m_1 = theta[m == 1]
p_m_1 = np.sum(m == 1) / m.size
ax = fig.add_subplot(gs[1, 1])
az.plot_posterior(theta_m_1, hdi_prob=.95, point_estimate='mode', ax=ax)
ax.set_title(f'$\\theta_{{m = 1}}. p(m = 1| D) = {p_m_1:.3f}$')
ax.set_xlabel('$\\theta$')

fig.tight_layout()


# -

# #### Using pseudo-priors to reduce autocorrelation

# +
def model_hier_2(y: jnp.ndarray):
    nb_obs = y.shape[0]

    model_prior = jnp.array([0.5, 0.5])
    m = numpyro.sample('m', dist.Categorical(model_prior))

    # Prior for theta 0 (corresponds to theta1 in the book).
    theta0 = numpyro.sample(
        'theta0', dist_utils.beta_dist_from_omega_kappa(.25, 12))

    # Prior for theta 1 (corresponds to theta2 in the book).
    theta1 = numpyro.sample(
        'theta1', dist_utils.beta_dist_from_omega_kappa(.75, 12))

    # Theta will be based on how the m is.
    theta = numpyro.deterministic('theta', jnp.where(m == 0, theta0, theta1))

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        numpyro.sample('y', dist.Bernoulli(theta), obs=y[idx])


kernel = DiscreteHMCGibbs(NUTS(model_hier_2))
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.r_[jnp.zeros(3), jnp.ones(6)],
)
mcmc.print_summary()
# -

idata = az.from_numpyro(mcmc)
az.plot_trace(idata)
plt.tight_layout()
