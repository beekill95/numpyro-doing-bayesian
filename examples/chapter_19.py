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

import arviz as az
from functools import reduce
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
from numpyro.infer import MCMC, NUTS
import numpyro_glm.metric.models as glm_metric
import pandas as pd
import seaborn as sns
from scipy.stats import norm

# # Chapter 19: Metric Predicted Variable with One Nominal Predictor
# ## Hierarchical Bayesian Approach

fruit_df = pd.read_csv('datasets/FruitflyDataReduced.csv')
fruit_df['CompanionNumber'] = fruit_df['CompanionNumber'].astype('category')
fruit_df.info()

fruit_df.describe()

# Run the model.

key = random.PRNGKey(0)
kernel = NUTS(glm_metric.one_nominal_predictor)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=10000)
mcmc.run(
    key,
    jnp.array(fruit_df['Longevity'].values),
    jnp.array(fruit_df['CompanionNumber'].cat.codes.values),
    len(fruit_df['CompanionNumber'].cat.categories))
mcmc.print_summary()

idata = az.from_numpyro(mcmc)
az.plot_trace(idata)
plt.tight_layout()

# Plot the posterior results.

# +
fig, ax = plt.subplots()
sns.swarmplot(fruit_df['CompanionNumber'], fruit_df['Longevity'], ax=ax)
ax.set_xlim(xmin=-1)

b0 = idata.posterior['b0'].values.flatten()
b_ = {name: idata.posterior['b_'].sel(b__dim_0=i).values.flatten()
      for i, name in enumerate(fruit_df['CompanionNumber'].cat.categories)}
ySigma = idata.posterior['ySigma'].values.flatten()

n_curves = 20
for i, name in enumerate(fruit_df['CompanionNumber'].cat.categories):
    indices = np.random.choice(
        len(idata.posterior.draw) * len(idata.posterior.chain), n_curves, replace=False)

    for idx in indices:
        rv = norm(b0[idx] + b_[name][idx], ySigma[idx])
        yrange = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 1000)
        xrange = rv.pdf(yrange)

        # Scale xrange.
        xrange = xrange * 0.75 / np.max(xrange)

        # Plot the posterior.
        ax.plot(-xrange + i, yrange)

fig.tight_layout()
