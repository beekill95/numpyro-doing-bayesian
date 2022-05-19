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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro_glm
import numpyro_glm.metric.models as glm_metric
from scipy.stats import norm

# # Chapter 17: Metric Predicted Variable with one Metric Predictor

# ## Simple Linear Regression

# +
x = np.random.uniform(-10, 10, size=500)
y = np.random.normal(10 + 2 * x, 2)

fig, ax = plt.subplots()
ax.scatter(x, y, c='black', s=4)
ax.set_title('Normal PDF around linear function')

xline = np.linspace(-10, 10, 1000)
ax.plot(xline, 10 + 2 * xline, lw=4, c='#87ceeb')

# TODO
for xinterval in [-7.5, -2.5, 2.5, 7.5]:
    y_ = np.linspace(xinterval - 6, xinterval + 6, 1000)
# -

# ## Robust Linear Regression

height_weight_30_data = pd.read_csv('datasets/HtWtData30.csv')
height_weight_30_data.describe()

# We will test the model with raw data (no standardization applied to both y and x).

mcmc_key = random.PRNGKey(0)
kernel = NUTS(glm_metric.one_metric_predictor_robust_no_standardization)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000)
mcmc.run(
    mcmc_key,
    jnp.array(height_weight_30_data.height.values),
    jnp.array(height_weight_30_data.weight.values),
)
mcmc.print_summary()

numpyro_glm.plot_diagnostic(mcmc, ['b0', 'b1', 'nu', 'sigma'])

# Here is the model with both y and x standardized.

mcmc_key = random.PRNGKey(0)
kernel = NUTS(glm_metric.one_metric_predictor_robust)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000)
mcmc.run(
    mcmc_key,
    jnp.array(height_weight_30_data.height.values),
    jnp.array(height_weight_30_data.weight.values),
)
mcmc.print_summary()

numpyro_glm.plot_diagnostic(mcmc, ['zb0', 'zb1', 'nu', 'zsigma', 'b0'])
