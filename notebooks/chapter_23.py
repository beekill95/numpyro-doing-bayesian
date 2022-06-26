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
import arviz as az
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median
import numpyro_glm.ordinal.models as glm_ordinal
import numpyro_glm.ordinal.plots as ordinal_plots
import pandas as pd
import seaborn as sns

numpyro.set_host_device_count(4)
# -

# # Chapter 23: Ordinal Predicted Variable
#
# ## The Case of A Single Group

ord_1_df = pd.read_csv('datasets/OrdinalProbitData-1grp-1.csv')
yord_1_cat = pd.CategoricalDtype([1, 2, 3, 4, 5, 6, 7], ordered=True)
ord_1_df['Y'] = ord_1_df['Y'].astype(yord_1_cat)

kernel = NUTS(glm_ordinal.yord_single_group,
              init_strategy=init_to_median,
              target_accept_prob=0.999)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(ord_1_df['Y'].cat.codes.values),
    K=yord_1_cat.categories.size,
)
mcmc.print_summary()

idata_yord_1 = az.from_numpyro(mcmc)
az.plot_trace(idata_yord_1)
plt.tight_layout()

# +
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 6))

# Plot latent score's mean posterior distribution.
ax = axes[0, 0]
az.plot_posterior(
    idata_yord_1,
    var_names='mu',
    hdi_prob=0.95,
    ref_val=2,
    point_estimate='mode',
    ax=ax)
ax.set_title('Mean')
ax.set_xlabel('$\mu$')

# Plot data with posterior distribution.
ax = axes[0, 1]
ax.hist(ord_1_df['Y'], bins=yord_1_cat.categories.size)

# Plot latent score's standard deviation.
ax = axes[1, 0]
az.plot_posterior(
    idata_yord_1,
    var_names='sigma',
    hdi_prob=0.95,
    point_estimate='mode',
    ax=ax)
ax.set_title('Std.')
ax.set_xlabel('$\sigma$')

# Plot effect size.
ax = axes[1, 1]
posterior = idata_yord_1.posterior
eff_size = (posterior['mu'] - 2) / posterior['sigma']
az.plot_posterior(
    eff_size,
    hdi_prob=0.95,
    point_estimate='mode',
    ref_val=0,
    ax=ax)
ax.set_title('Effect Size')
ax.set_xlabel('$(\mu - 2) / \sigma$')

# Plot thresholds.
ax = axes[2, 0]
ordinal_plots.plot_threshold_scatter(
    idata_yord_1,
    ['thres_1', 'thres_2', 'thres_3', 'thres_4', 'thres_5', 'thres_6'],
    ax=ax)

axes[2, 1].remove()
fig.tight_layout()
# -

ord_2_df = pd.read_csv('datasets/OrdinalProbitData-1grp-2.csv')
yord_2_cat = pd.CategoricalDtype([1, 2, 3, 4, 5, 6, 7], ordered=True)
ord_2_df['Y'] = ord_2_df['Y'].astype(yord_2_cat)

kernel = NUTS(glm_ordinal.yord_single_group,
              init_strategy=init_to_median,
              target_accept_prob=0.99)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(ord_2_df['Y'].cat.codes.values),
    K=yord_2_cat.categories.size,
)
mcmc.print_summary()

idata_yord_2 = az.from_numpyro(mcmc)
az.plot_trace(idata_yord_2)
plt.tight_layout()
