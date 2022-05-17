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
#     display_name: doing_bayes
#     language: python
#     name: doing_bayes
# ---

# %cd ..
# %load_ext autoreload
# %autoreload 2

import arviz as az
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro_glm.metric.models as glm_metric
from scipy.stats import norm

# # Chapter 16: Metric-Predicted Variable on One or Two Groups

# ## Estimating Mean and Standard Deviation of a Normal distribution
#
# ### Create synthesis data.

# +
MEAN = 5
STD = 3
N = 100

y = np.random.normal(MEAN, STD, N)

# Plot the histogram and true PDF of normal distribution.
fig, ax = plt.subplots()
ax.hist(y, density=True, label='Histogram of $y$')
xmin, xmax = ax.get_xlim()
p = np.linspace(xmin, xmax, 1000)
pdf = np.exp(-(p - MEAN)**2 / (2 * STD**2)) / (STD * np.sqrt(2 * np.pi))
ax.plot(p, pdf, label='Normal PDF')
ax.legend()
fig.tight_layout()
# -

# ### Metric Model

numpyro.render_model(
    glm_metric.one_group,
    model_args=(jnp.array(y),),
    render_params=True
)

# Now, we'll try to apply the metric model on that data to see
# if it can recover the parameter.

mcmc_key = random.PRNGKey(0)
kernel = NUTS(glm_metric.one_group)
mcmc = MCMC(kernel, num_warmup=250, num_samples=750)
mcmc.run(mcmc_key, jnp.array(y))
mcmc.print_summary()

# Plot diagnostics plot to see if the MCMC chains are well-behaved.

# +
idata = az.from_numpyro(mcmc)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

# Plot posterior mean.
ax = axes[0, 0]
ax.set_title('Mean')
az.plot_posterior(idata, var_names=['mean'], point_estimate='mode', kind='hist', hdi_prob=0.95, ax=ax)
ax.set_xlabel('$\mu$')

# Plot data with posterior.
ax = axes[0, 1]
ax.set_title('Data w. posterior pred.')
ax.hist(y, density=True)

## Plot some posterior distributions.
n_curves = 20
samples_idx = np.random.choice(len(idata.posterior.chain) * len(idata.posterior.draw), n_curves, replace=False)
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 1000)
for idx in samples_idx:
    sample_mean = idata.posterior['mean'].values.flatten()[idx]
    sample_std = idata.posterior['std'].values.flatten()[idx]
    ax.plot(x, norm.pdf(x, sample_mean, sample_std), c='#87ceeb')

ax.set_xlabel('y')

# Plot standard deviation.
ax = axes[1, 0]
ax.set_title('Standard Deviation')
az.plot_posterior(idata, var_names=['std'], point_estimate='mode', kind='hist', hdi_prob=0.95, ax=ax)
ax.set_xlabel('$\sigma$')

# Remove effect size plot.
ax = axes[1, 1]
ax.remove()

fig.tight_layout()
