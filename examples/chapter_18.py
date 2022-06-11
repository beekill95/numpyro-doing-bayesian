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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs, HMCECS
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

df_SAT[x_SAT_names].corr()
# -

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
# -

# ## Multiplicative Interaction of Metric Predictors

# +
df_SAT['Spend_Prcnt'] = df_SAT['Spend'] * df_SAT['PrcntTake']
x_SAT_names = ['Spend', 'PrcntTake', 'Spend_Prcnt']
x_SAT_multiplicative = df_SAT[x_SAT_names].values

df_SAT[x_SAT_names].corr()
# -

key = random.PRNGKey(0)
model = NUTS(glm_metric.multi_metric_predictors_robust)
mcmc = MCMC(model, num_warmup=1000, num_samples=20000)
mcmc.run(key, y_SAT, x_SAT_multiplicative)
mcmc.print_summary()

# +
idata_SAT_multiplicative = az.from_numpyro(
    mcmc,
    coords=dict(predictors=[0, 1, 2]),
    dims=dict(b_=['predictors']))
posterior_SAT_multiplicative = idata_SAT_multiplicative.posterior

fig_SAT_multiplicative_posteriors, axes = plt.subplots(
    nrows=2, ncols=3, figsize=(12, 8))
to_plot_posteriors_SAT_multiplicative = {
    'Intercept': posterior_SAT_multiplicative['b0'].values.flatten(),
    'Spend Coeff': posterior_SAT_multiplicative['b_'].sel(dict(predictors=0)).values.flatten(),
    'PrcntTake Coeff': posterior_SAT_multiplicative['b_'].sel(dict(predictors=1)).values.flatten(),
    'Spend_Prcnt Coeff': posterior_SAT_multiplicative['b_'].sel(dict(predictors=2)).values.flatten(),
    'Scale': posterior_SAT_multiplicative['sigma'].values.flatten(),
    'Log Normality': np.log(posterior_SAT_multiplicative['nu'].values.flatten()),
}

for ax, (title, values) in zip(axes.flatten(), to_plot_posteriors_SAT_multiplicative.items()):
    az.plot_posterior(values, point_estimate='mode', hdi_prob=0.95, ax=ax)
    ax.set_title(title)

fig_SAT_multiplicative_posteriors.tight_layout()

# +
fig_SAT_slopes, axes = plt.subplots(nrows=2, figsize=(12, 8))

# Spend as a function of Percent Take.
ax = axes[0]
percent_take = np.linspace(4, 80, 20)
spend_slopes = (to_plot_posteriors_SAT_multiplicative['Spend Coeff'].reshape(-1, 1)
                + to_plot_posteriors_SAT_multiplicative['Spend_Prcnt Coeff'].reshape(-1, 1) * percent_take.reshape(1, -1))
spend_hdis = az.hdi(spend_slopes, hdi_prob=0.95)
spend_medians = np.median(spend_slopes, axis=0)

ax.errorbar(percent_take, spend_medians,
            yerr=spend_hdis[:, 0] - spend_hdis[:, 1], fmt='o', label='Spend Slope Median')
ax.set_title('Spend Slope vs. Percent Take')
ax.set_xlabel('Percent Take')
ax.set_ylabel('Spend Slope')
ax.legend()

# Percent Take as a function of Spend.
ax = axes[1]
spend = np.linspace(3, 10, 20)
prcnt_slopes = (to_plot_posteriors_SAT_multiplicative['PrcntTake Coeff'].reshape(-1, 1)
                + to_plot_posteriors_SAT_multiplicative['Spend_Prcnt Coeff'].reshape(-1, 1) * spend.reshape(1, -1))
prcnt_hdis = az.hdi(prcnt_slopes, hdi_prob=0.95)
prcnt_medians = np.median(prcnt_slopes, axis=0)

ax.errorbar(spend, prcnt_medians,
            yerr=prcnt_hdis[:, 0] - prcnt_hdis[:, 1], fmt='o', label='Percent Take Slope Median')
ax.set_title('Percent Take Slope vs. Spend')
ax.set_xlabel('Spend')
ax.set_ylabel('Percent Take Slope')
ax.legend()

fig_SAT_slopes.tight_layout()
# -

# ## Shrinkage of Regression Coefficients
# ### Without Shrinkage Model


# +
def normalize(values):
    return (values - np.mean(values)) / np.std(values)


nb_random_preds = 12

df_SAT_random = df_SAT.copy()
for i in range(nb_random_preds):
    df_SAT_random[f'xRand{i}'] = normalize(
        np.random.normal(0, 1, size=len(df_SAT)))

df_SAT_random.describe()

# +
x_SAT_random_cols = ['Spend', 'PrcntTake',
                     *(f'xRand{i}' for i in range(nb_random_preds))]

x_SAT_random = df_SAT_random[x_SAT_random_cols].values
y_SAT_random = df_SAT_random['SATT'].values

key = random.PRNGKey(0)
kernel = NUTS(glm_metric.multi_metric_predictors_robust)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000)
mcmc.run(key, y_SAT_random, x_SAT_random)
mcmc.print_summary()

# +
idata_SAT_random = az.from_numpyro(
    mcmc,
    coords=dict(predictors=list(range(14))),
    dims=dict(b_=['predictors']))
posterior_SAT_random = idata_SAT_random.posterior

fig_SAT_random_posteriors, axes = plt.subplots(
    nrows=4, ncols=3, figsize=(12, 8))
to_plot_posteriors_SAT_random = {
    'Intercept': posterior_SAT_random['b0'].values.flatten(),
    'Spend Coeff': posterior_SAT_random['b_'].sel(dict(predictors=0)).values.flatten(),
    'PrcntTake Coeff': posterior_SAT_random['b_'].sel(dict(predictors=1)).values.flatten(),
    **{f'xRand{i} Coeff': posterior_SAT_random['b_'].sel(dict(predictors=i + 2)).values.flatten()
        for i in [0, 1, 2, 9, 10, 11]},
    'Scale': posterior_SAT_random['sigma'].values.flatten(),
    'Log Normality': np.log(posterior_SAT_random['nu'].values.flatten()),
}

for ax, (title, values) in zip(axes.flatten(), to_plot_posteriors_SAT_random.items()):
    az.plot_posterior(values, point_estimate='mode', hdi_prob=0.95, ax=ax)
    ax.set_title(title)

fig_SAT_random_posteriors.tight_layout()
# -

# ### Shrinkage Model

key = random.PRNGKey(0)
kernel = NUTS(glm_metric.multi_metric_predictors_robust_with_shrinkage)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000)
mcmc.run(key, y_SAT_random, x_SAT_random)
mcmc.print_summary()

# +
idata_SAT_random_shrinkage = az.from_numpyro(
    mcmc,
    coords=dict(predictors=list(range(14))),
    dims=dict(b_=['predictors']))
posterior_SAT_random_shrinkage = idata_SAT_random_shrinkage.posterior

fig_SAT_random_posteriors_shrinkage, axes = plt.subplots(
    nrows=4, ncols=3, figsize=(12, 8))
to_plot_posteriors_SAT_random_shrinkage = {
    'Intercept': posterior_SAT_random_shrinkage['b0'].values.flatten(),
    'Spend Coeff': posterior_SAT_random_shrinkage['b_'].sel(dict(predictors=0)).values.flatten(),
    'PrcntTake Coeff': posterior_SAT_random_shrinkage['b_'].sel(dict(predictors=1)).values.flatten(),
    **{f'xRand{i} Coeff': posterior_SAT_random_shrinkage['b_'].sel(dict(predictors=i + 2)).values.flatten()
        for i in [0, 1, 2, 9, 10, 11]},
    'Scale': posterior_SAT_random_shrinkage['sigma'].values.flatten(),
    'Log Normality': np.log(posterior_SAT_random_shrinkage['nu'].values.flatten()),
}

for ax, (title, values) in zip(axes.flatten(), to_plot_posteriors_SAT_random_shrinkage.items()):
    az.plot_posterior(values, point_estimate='mode', hdi_prob=0.95, ax=ax)
    ax.set_title(title)

fig_SAT_random_posteriors_shrinkage.tight_layout()
# -

# ## Variable Selection

# +
x_SAT_all_names = ['Spend', 'StuTeaRat', 'Salary', 'PrcntTake']
x_SAT_all = df_SAT[x_SAT_all_names].values

key = random.PRNGKey(0)
kernel = DiscreteHMCGibbs(
    NUTS(glm_metric.multi_metric_predictors_robust_with_selection), modified=True)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=40000)
mcmc.run(key, y_SAT, x_SAT_all)
mcmc.print_summary()

# +
idata_SAT_all_with_selection = az.from_numpyro(
    mcmc,
    coords=dict(predictors=list(range(4))),
    dims=dict(b_=['predictors'], zb=['predictors'], delta=['predictors']),
)
posterior_SAT_all_with_selection = idata_SAT_all_with_selection.posterior


def plot_selected_model_posteriors(predictors):
    PREDICTORS_NAME = {i: name for i, name in enumerate(x_SAT_all_names)}

    # Mask to differentiate which coefficients values should be included
    # in the posterior plot.
    mask = reduce(
        lambda acc, p: acc & (posterior_SAT_all_with_selection['delta'].sel(
            predictors=p).values == (1 if p in predictors else 0)),
        list(PREDICTORS_NAME.keys())[1:],
        posterior_SAT_all_with_selection['delta'].sel(predictors=0).values == (1 if 0 in predictors else 0))
    mask = mask.astype(bool)

    # Calculate the model's probability.
    model_prob = mask.sum() / np.prod(mask.shape)

    # Create figure.
    fig, axes = plt.subplots(
        ncols=len(PREDICTORS_NAME.keys()) + 1, figsize=(15, 4))
    fig.suptitle(f'Model Prob = {model_prob:.3f}')
    axes = axes.flatten()

    # Plot the posterior of the intercept.
    ax = axes[0]
    az.plot_posterior(
        posterior_SAT_all_with_selection['b0'].values[mask],
        point_estimate='mode',
        hdi_prob=0.95,
        ax=ax)
    ax.set_title('Intercept')

    # Plot the posterior of the coefficients.
    for predictor, ax in zip(PREDICTORS_NAME.keys(), axes[1:]):
        if predictor in predictors:
            az.plot_posterior(
                posterior_SAT_all_with_selection['b_'].sel(
                    predictors=predictor).values[mask],
                point_estimate='mode',
                hdi_prob=0.95,
                ax=ax)
            ax.set_title(PREDICTORS_NAME[predictor])
        else:
            ax.remove()

    fig.tight_layout()


models_to_plot = [
    [0, 3],
    [3],
    [2, 3],
    [1, 2, 3],
    [0, 2, 3],
    [0, 1, 3],
    [1, 3],
    [0, 1, 2, 3],
]

for model in models_to_plot:
    plot_selected_model_posteriors(model)
