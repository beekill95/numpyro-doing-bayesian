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

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro_glm.metric.models as glm_metric
import numpyro_glm.metric.plots as plots
import pandas as pd
from scipy.stats import norm, t

# # Chapter 16: Metric-Predicted Variable on One or Two Groups

# ## Estimating Mean and Standard Deviation of a Normal distribution

# ### Metric Model

numpyro.render_model(
    glm_metric.one_group,
    model_args=(jnp.ones(5), ),
    render_params=True)

# ### Synthesis data

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
ax.plot(p, norm.pdf(p, loc=MEAN, scale=STD), label='Normal PDF')
ax.legend()
fig.tight_layout()
# -

# Now, we'll try to apply the metric model on that data to see
# if it can recover the parameter.

mcmc_key = random.PRNGKey(0)
kernel = NUTS(glm_metric.one_group)
mcmc = MCMC(kernel, num_warmup=250, num_samples=750)
mcmc.run(mcmc_key, jnp.array(y))
mcmc.print_summary()

# Plot diagnostics plot to see if the MCMC chains are well-behaved.

fig = plots.plot_st(mcmc, y)

# ### Smart Group IQ

iq_data = pd.read_csv('datasets/TwoGroupIQ.csv')
smart_group_data = iq_data[iq_data.Group == 'Smart Drug']
smart_group_data.describe()

# Then, we will apply the one group model to the data
# and plot the results.

mcmc_key = random.PRNGKey(0)
kernel = NUTS(glm_metric.one_group)
mcmc = MCMC(kernel, num_warmup=250, num_samples=750)
mcmc.run(mcmc_key, jnp.array(smart_group_data.Score.values))
mcmc.print_summary()

fig = plots.plot_st(
    mcmc, smart_group_data.Score.values,
    mean_comp_val=100,
    std_comp_val=15,
    effsize_comp_val=0,
)

# ## Outliers and Robust Estimation: $t$ Distribution

# ### Robust Metric Model

numpyro.render_model(
    glm_metric.one_group_robust,
    model_args=(jnp.ones(5),),
    render_params=True)

# ### Synthesis Data

# +
MEAN = 5
SIGMA = 3
NORMALITY = 3
N = 1000

y = np.random.standard_t(NORMALITY, size=N) * SIGMA + MEAN

# Plot the histogram and true PDF of normal distribution.
fig, ax = plt.subplots()
ax.hist(y, density=True, bins=100, label='Histogram of $y$')
xmin, xmax = ax.get_xlim()
p = np.linspace(xmin, xmax, 1000)
ax.plot(p, t.pdf(p, loc=MEAN, scale=SIGMA, df=NORMALITY), label='Student-$t$ PDF')
ax.plot(p, norm.pdf(p, loc=y.mean(), scale=y.std()), label='Normal PDF using\ndata mean and std')
ax.legend()
fig.tight_layout()
# -

# Using the robust metric model on our synthesis data
# to see if it can recover the original parameters.

mcmc_key = random.PRNGKey(0)
kernel = NUTS(glm_metric.one_group_robust)
mcmc = MCMC(kernel, num_warmup=250, num_samples=750)
mcmc.run(mcmc_key, jnp.array(y))
mcmc.print_summary()

fig = plots.plot_st_2(mcmc, y)

# ### Smart Drug Group Data

mcmc_key = random.PRNGKey(0)
kernel = NUTS(glm_metric.one_group_robust)
mcmc = MCMC(kernel, num_warmup=250, num_samples=750)
mcmc.run(mcmc_key, jnp.array(smart_group_data.Score.values))
mcmc.print_summary()

fig = plots.plot_st_2(
    mcmc, smart_group_data.Score.values,
    mean_comp_val=100,
    sigma_comp_val=15,
    effsize_comp_val=0)
