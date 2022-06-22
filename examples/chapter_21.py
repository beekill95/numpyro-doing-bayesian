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
import numpyro_glm.logistic.models as glm_logistic
import pandas as pd
import seaborn as sns

numpyro.set_host_device_count(4)
# -

# # Chapter 21: Dichotomous Predicted Variable
# ## Multiple Metric Predictors

htwt_df = pd.read_csv('datasets/HtWtData110.csv')
htwt_df.info()

# ### One Predictor: Weight

kernel = NUTS(glm_logistic.dich_multi_metric_predictors)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(htwt_df['male'].values),
    x=jnp.array(htwt_df[['weight']].values),
)
mcmc.print_summary()

idata_wt = az.from_numpyro(
    mcmc,
    dims=dict(preds=['weight']),
    coords=dict(b=['preds']))
az.plot_trace(idata_wt, ['b0', 'b'])
plt.tight_layout()

# +
n_curves = 20
posterior = idata_wt.posterior
curve_indices = np.random.choice(
    posterior.draw.size * posterior.chain.size,
    n_curves,
    replace=False
)
b0 = posterior['b0'].values.flatten()
b1 = posterior['b'].values.flatten()

fig, ax = plt.subplots()
ax.plot(htwt_df['weight'], htwt_df['male'], 'o')
wt_range = np.linspace(*ax.get_xlim(), 1000)

for idx in curve_indices:
    y = b0[idx] + b1[idx] * wt_range
    y = 1. / (1 + np.exp(-y))
    ax.plot(wt_range, y)

fig.tight_layout()
# -

# ### Two Predictors: Weight and Height

kernel = NUTS(glm_logistic.dich_multi_metric_predictors)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(htwt_df['male'].values),
    x=jnp.array(htwt_df[['height', 'weight']].values),
)
mcmc.print_summary()

idata_htwt = az.from_numpyro(
    mcmc,
    coords=dict(pred=['height', 'weight']),
    dims=dict(b=['pred']))
az.plot_trace(idata_htwt, ['b0', 'b'])
plt.tight_layout()

# +
n_curves = 20
posterior = idata_htwt.posterior
curve_indices = np.random.choice(
    posterior.draw.size * posterior.chain.size,
    n_curves,
    replace=False
)
b0 = posterior['b0'].values.flatten()
b1 = posterior['b'].sel(pred='height').values.flatten()
b2 = posterior['b'].sel(pred='weight').values.flatten()

fig, ax = plt.subplots()
sns.scatterplot(
    x='weight', y='height',
    style='male',
    hue='male',
    markers=True,
    data=htwt_df,
    ax=ax)
wt_range = np.linspace(*ax.get_xlim(), 1000)
ht_range = np.linspace(*ax.get_ylim(), 1000)
wt, ht = np.meshgrid(wt_range, ht_range, indexing='xy')

for idx in curve_indices:
    lin_core = b0[idx] + b1[idx] * ht + b2[idx] * wt
    prob = 1. / (1 + np.exp(-lin_core))
    ax.contour(wt, ht, prob, alpha=0.6, levels=[0.5])

fig.tight_layout()
