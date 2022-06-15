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

idata = az.from_numpyro(
    mcmc,
    coords=dict(CompanionNumber=fruit_df['CompanionNumber'].cat.categories),
    dims=dict(b_=['CompanionNumber']))
az.plot_trace(idata)
plt.tight_layout()

# Plot the posterior results.

# +
fig, ax = plt.subplots()
sns.swarmplot(fruit_df['CompanionNumber'], fruit_df['Longevity'], ax=ax)
ax.set_xlim(xmin=-1)

b0 = idata.posterior['b0'].values.flatten()
b_ = {name: idata.posterior['b_'].sel(CompanionNumber=name).values.flatten()
      for name in fruit_df['CompanionNumber'].cat.categories}
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


# -

# Plot contrasts to compare between groups.

# +
def plot_contrasts(idata: az.InferenceData, contrasts: 'list[dict]', figsize=(10, 6)):
    fig, axes = plt.subplots(nrows=2, ncols=len(contrasts), figsize=figsize)
    posterior = idata.posterior

    def mean(values: list):
        return sum(values) / len(values)

    for i, contrast in enumerate(contrasts):
        plot_title = f'{".".join(contrast["left"])}\nvs\n{".".join(contrast["right"])}'

        # Plot difference.
        ax = axes[0, i]
        left = mean([posterior['b_'].sel(CompanionNumber=n).values
                     for n in contrast['left']])
        right = mean([posterior['b_'].sel(CompanionNumber=n).values
                      for n in contrast['right']])
        diff = left - right

        az.plot_posterior(
            diff, hdi_prob=0.95,
            point_estimate='mode',
            ref_val=contrast['refVal'], rope=contrast['rope'],
            ax=ax)
        ax.set_title(plot_title)
        ax.set_xlabel('Difference')

        # Plot effect size.
        ax = axes[1, i]
        effSize = diff / posterior['ySigma']

        az.plot_posterior(
            effSize, hdi_prob=0.95,
            point_estimate='mode',
            ref_val=contrast['effSizeRefVal'], rope=contrast['effSizeRope'],
            ax=ax)
        ax.set_title(plot_title)
        ax.set_xlabel('Effect Size')

    fig.tight_layout()
    return fig


contrasts = [
    dict(left=['Pregnant1', 'Pregnant8'], right=['None0'],
         refVal=0.0, rope=(-1.5, 1.5),
         effSizeRefVal=0.0, effSizeRope=(-0.1, 0.1)),
    dict(left=['Pregnant1', 'Pregnant8', 'None0'],
         right=['Virgin1', 'Virgin8'],
         refVal=0.0, rope=(-1.5, 1.5),
         effSizeRefVal=0.0, effSizeRope=(-0.1, 0.1)),
    dict(left=['Pregnant1', 'Pregnant8', 'None0'], right=['Virgin1'],
         refVal=0.0, rope=(-1.5, 1.5),
         effSizeRefVal=0.0, effSizeRope=(-0.1, 0.1)),
    dict(left=['Virgin1'], right=['Virgin8'],
         refVal=0.0, rope=(-1.5, 1.5),
         effSizeRefVal=0.0, effSizeRope=(-0.1, 0.1)),
]

_ = plot_contrasts(idata, contrasts, figsize=(15, 6))
