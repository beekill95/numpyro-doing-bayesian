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
from numpyro.infer import MCMC, NUTS
import numpyro_glm
import numpyro_glm.metric.models as glm_metric
from scipy.stats import pearsonr

# # Chapter 18: Metric Predicted Variable with Multiple Metric Predictors
# ## Multiple Linear Regression

df_SAT = pd.read_csv('datasets/Guber1999data.csv')
df_SAT.info()

df_SAT.describe()

# +
y_SAT = df_SAT.SATT.values
x_SAT_names = ['Spend', 'PrcntTake']
x_SAT = df_SAT[x_SAT_names].values

key = random.PRNGKey(0)
model = NUTS(glm_metric.multi_metric_predictors_robust)
mcmc = MCMC(model, num_warmup=1000, num_samples=20000)
mcmc.run(key, y_SAT, x_SAT)
mcmc.print_summary()

# +
idata_SAT = az.from_numpyro(
    mcmc,
    coords=dict(predictors=[0, 1]),
    dims=dict(b_=['predictors']))
posterior_SAT = idata_SAT.posterior

fig_SAT_posteriors, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
to_plot_posteriors_SAT = {
    'Intercept': posterior_SAT['b0'].values.flatten(),
    'Spend Coeff': posterior_SAT['b_'].sel(dict(predictors=0)).values.flatten(),
    'PrcntTake Coeff': posterior_SAT['b_'].sel(dict(predictors=1)).values.flatten(),
    'Scale': posterior_SAT['sigma'].values.flatten(),
    'Log Normality': np.log(posterior_SAT['nu'].values.flatten()),
}

for ax, (title, values) in zip(axes.flatten(), to_plot_posteriors_SAT.items()):
    az.plot_posterior(values, point_estimate='mode', hdi_prob=0.95, ax=ax)
    ax.set_title(title)

fig_SAT_posteriors.tight_layout()

# +
fig_SAT_pairwise_scatter, axes = plt.subplots(
    nrows=5, ncols=5, figsize=(20, 20))

for ith, ith_var in enumerate(to_plot_posteriors_SAT.keys()):
    for jth, jth_var in enumerate(to_plot_posteriors_SAT.keys()):
        ax = axes[jth, ith]

        if ith == jth:
            numpyro_glm.plot_text(ith_var, ax)
        elif ith < jth:
            ith_var_data = to_plot_posteriors_SAT[ith_var]
            jth_var_data = to_plot_posteriors_SAT[jth_var]

            corr, _ = pearsonr(ith_var_data, jth_var_data)
            numpyro_glm.plot_text(f'{corr:.2f}', ax)
        else:
            ith_var_data = to_plot_posteriors_SAT[ith_var]
            jth_var_data = to_plot_posteriors_SAT[jth_var]

            ax.scatter(ith_var_data, jth_var_data)

fig_SAT_pairwise_scatter.tight_layout()
