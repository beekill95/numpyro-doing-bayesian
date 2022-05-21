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

# ## Hierarchical Regression on Individuals within Groups

hier_linear_reg_data = pd.read_csv('datasets/HierLinRegressData.csv')
hier_linear_reg_data['Subj'] = hier_linear_reg_data['Subj'].astype('category')
hier_linear_reg_data.describe()

mcmc_key = random.PRNGKey(0)
kernel = NUTS(glm_metric.hierarchical_one_metric_predictor_multi_groups_robust)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000)
mcmc.run(
    mcmc_key,
    jnp.array(hier_linear_reg_data.Y.values),
    jnp.array(hier_linear_reg_data.X.values),
    jnp.array(hier_linear_reg_data.Subj.cat.codes.values),
    len(hier_linear_reg_data.Subj.cat.categories),
)
mcmc.print_summary()

# ## Quadratic Trend and Weighted Data

income_data_3yr = pd.read_csv('datasets/IncomeFamszState3yr.csv', skiprows=1)
income_data_3yr['State'] = income_data['State'].astype('category')
income_data_3yr.describe()

# +
fig, ax = plt.subplots()

for state in income_data_3yr['State'].unique():
    state_data = income_data_3yr[income_data_3yr['State'] == state]
    state_data = state_data.sort_values('FamilySize')

    ax.plot(state_data['FamilySize'], state_data['MedianIncome'], 'o-')

ax.set_title('Median Income of Various States')
ax.set_xlabel('Family Size')
ax.set_ylabel('Median Income')
fig.tight_layout()
# -

mcmc_key = random.PRNGKey(0)
kernel = NUTS(
    glm_metric.hierarchical_quadtrend_one_metric_predictor_multi_groups_robust)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000)
mcmc.run(
    mcmc_key,
    jnp.array(income_data_3yr['MedianIncome'].values),
    jnp.array(income_data_3yr['FamilySize'].values),
    jnp.array(income_data_3yr['State'].cat.codes.values),
    len(income_data_3yr['State'].cat.categories),
    jnp.array(income_data_3yr['SampErr'].values),
)
mcmc.print_summary()

# +
idata = az.from_numpyro(mcmc)
posterior = idata.posterior

n_posterior_curves = 20
x = np.linspace(1, 7.5, 1000)
curves = np.random.choice(
    len(posterior.draw) * len(posterior.chain), n_posterior_curves, replace=False)

for curve in curves:
    b0_mean = posterior['b0_mean'].values.flatten()[curve]
    b1_mean = posterior['b1_mean'].values.flatten()[curve]
    b2_mean = posterior['b2_mean'].values.flatten()[curve]

    ax.plot(x, b0_mean + b1_mean * x + b2_mean * x**2, c='b')

fig

# +
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

for state_idx, state, ax in zip(income_data_3yr['State'].cat.codes,
                                income_data_3yr['State'].cat.categories,
                                axes.flatten()):
    # Superimpose posterior curves.
    for curve in curves:
        b0 = (posterior['b0']
              .sel(dict(b0_dim_0=state_idx)).values.flatten()[curve])
        b1 = (posterior['b1']
              .sel(dict(b1_dim_0=state_idx)).values.flatten()[curve])
        b2 = (posterior['b2']
              .sel(dict(b2_dim_0=state_idx)).values.flatten()[curve])
        ax.plot(x, b0 + b1 * x + b2 * x**2, c='b')

    # Plot state median income.
    state_data = income_data_3yr[income_data_3yr['State'] == state]
    state_data = state_data.sort_values('FamilySize')
    ax.plot(state_data['FamilySize'], state_data['MedianIncome'], 'ko')

    ax.set_title(state)

fig.tight_layout()
