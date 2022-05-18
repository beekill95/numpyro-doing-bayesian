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
