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
import numpyro_glm
import numpyro_glm.logistic.models as glm_logistic
import pandas as pd
from scipy.special import expit
from scipy.stats import beta
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
# -

# ## Robust Logistic Regression

kernel = NUTS(glm_logistic.dich_multi_metric_predictors_robust)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(htwt_df['male'].values),
    x=jnp.array(htwt_df[['weight']].values),
)
mcmc.print_summary()

idata_wt_robust = az.from_numpyro(
    mcmc,
    coords=dict(preds=['weight']),
    dims=dict(b=['preds']))
az.plot_trace(idata_wt_robust, ['b0', 'b'])
plt.tight_layout()

# +
n_curves = 20
posterior = idata_wt_robust.posterior
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

_ = numpyro_glm.plot_pairwise_scatter(
    mcmc, ['b0', 'b', 'guess'])

# ## Nominal Predictors
# ### Multiple Groups
# #### Homogeneous Concentration

batting_df = pd.read_csv('datasets/BattingAverage.csv')
batting_df['PriPos'] = batting_df['PriPos'].astype('category')
batting_df['Player'] = batting_df['Player'].astype('category')
batting_df.info()

kernel = NUTS(glm_logistic.binom_one_nominal_predictor,
              init_strategy=init_to_median)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(batting_df['Hits'].values),
    grp=jnp.array(batting_df['PriPos'].cat.codes.values),
    N=jnp.array(batting_df['AtBats'].values),
    nb_groups=batting_df['PriPos'].cat.categories.size,
)
mcmc.print_summary()

idata_binom = az.from_numpyro(
    mcmc,
    coords=dict(pos=batting_df['PriPos'].cat.categories.values),
    dims=dict(b=['pos'], omega=['pos']),
)
# az.plot_trace(idata_binom)
# plt.tight_layout()

# Plot predicted posterior distribution with data.

# +
fig, ax = plt.subplots(figsize=(12, 4))
batting_df['Hits / AtBats'] = batting_df['Hits'] / batting_df['AtBats']
sns.stripplot(x='PriPos', y='Hits / AtBats', data=batting_df, ax=ax)

posterior = idata_binom.posterior
kappa = posterior['kappa'].values.flatten()
n_curves = 20

for i, pri_pos in enumerate(batting_df['PriPos'].cat.categories):
    omega = posterior['omega'].sel(pos=pri_pos).values.flatten()
    curve_indices = np.random.choice(
        posterior.chain.size * posterior.draw.size,
        n_curves,
        replace=False,
    )

    for idx in curve_indices:
        rv = beta(omega[idx] * (kappa[idx] - 2) + 1,
                  (1 - omega[idx]) * (kappa[idx] - 2) + 1)
        yrange = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 1000)
        xpdf = rv.pdf(yrange)
        xpdf *= 0.75 / np.max(xpdf)

        ax.plot(i - xpdf, yrange, c='b', alpha=.1)

fig.tight_layout()

# +
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

differences = [
    ('b', 'Pitcher', 'Catcher'),
    ('b', 'Catcher', '1st Base'),
    ('omega', 'Pitcher', 'Catcher'),
    ('omega', 'Catcher', '1st Base'),
]
for ax, (var, left_pos, right_pos) in zip(axes.flatten(), differences):
    left_val = posterior[var].sel(pos=left_pos).values
    right_val = posterior[var].sel(pos=right_pos).values
    diff = left_val - right_val

    az.plot_posterior(diff, point_estimate='mode',
                      hdi_prob=0.95, ref_val=0, ax=ax)
    ax.set_title(f'{var}: {left_pos} vs {right_pos}')
    ax.set_xlabel(f'Difference (in {var})')

fig.tight_layout()
# -

# #### Heterogeneous Concentration (Exercise 21.3)

kernel = NUTS(glm_logistic.binom_one_nominal_predictor_het,
              init_strategy=init_to_median)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(batting_df['Hits'].values),
    grp=jnp.array(batting_df['PriPos'].cat.codes.values),
    N=jnp.array(batting_df['AtBats'].values),
    nb_groups=batting_df['PriPos'].cat.categories.size,
)
mcmc.print_summary()

idata_binom_het = az.from_numpyro(
    mcmc,
    coords=dict(pos=batting_df['PriPos'].cat.categories.values),
    dims=dict(b=['pos'], omega=['pos'], kappa=['pos']),
)
# az.plot_trace(idata_binom)
# plt.tight_layout()

# Plot predicted posterior distribution with data.

# +
fig, ax = plt.subplots(figsize=(12, 4))
batting_df['Hits / AtBats'] = batting_df['Hits'] / batting_df['AtBats']
sns.stripplot(x='PriPos', y='Hits / AtBats', data=batting_df, ax=ax)

posterior = idata_binom_het.posterior
n_curves = 20

for i, pri_pos in enumerate(batting_df['PriPos'].cat.categories):
    omega = posterior['omega'].sel(pos=pri_pos).values.flatten()
    kappa = posterior['kappa'].sel(pos=pri_pos).values.flatten()

    curve_indices = np.random.choice(
        posterior.chain.size * posterior.draw.size,
        n_curves,
        replace=False,
    )

    for idx in curve_indices:
        rv = beta(omega[idx] * (kappa[idx] - 2) + 1,
                  (1 - omega[idx]) * (kappa[idx] - 2) + 1)
        yrange = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 1000)
        xpdf = rv.pdf(yrange)
        xpdf *= 0.75 / np.max(xpdf)

        ax.plot(i - xpdf, yrange, c='b', alpha=.1)

fig.tight_layout()

# +
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

differences = [
    ('b', 'Pitcher', 'Catcher'),
    ('b', 'Catcher', '1st Base'),
    ('omega', 'Pitcher', 'Catcher'),
    ('omega', 'Catcher', '1st Base'),
]
for ax, (var, left_pos, right_pos) in zip(axes.flatten(), differences):
    left_val = posterior[var].sel(pos=left_pos).values
    right_val = posterior[var].sel(pos=right_pos).values
    diff = left_val - right_val

    az.plot_posterior(diff, point_estimate='mode',
                      hdi_prob=0.95, ref_val=0, ax=ax)
    ax.set_title(f'{var}: {left_pos} vs {right_pos}')
    ax.set_xlabel(f'Difference (in {var})')

fig.tight_layout()
