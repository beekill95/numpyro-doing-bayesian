# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     notebook_metadata_filter: title, author
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   title: '[Doing Bayesian Data Analysis] Chapter 20: Metric Predicted Variable with Multiple Nominal Predictors'
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
from numpyro.infer.initialization import init_to_median
from numpyro.infer import NUTS, MCMC
import numpyro_glm.metric.models as glm_metric
import pandas as pd
import seaborn as sns
from scipy.stats import norm, t

numpyro.set_host_device_count(2)
# -

# # Chapter 20: Metric Predicted Variable with Multiple Nominal Predictors
# ## Hierarchical Bayesian Approach

salary_df = pd.read_csv('datasets/Salary.csv')
salary_df['Org'] = salary_df['Org'].astype('category')
salary_df['Pos'] = (salary_df['Pos']
                    .astype('category').cat
                    .set_categories(['FT3', 'FT2', 'FT1', 'NDW', 'DST']).cat
                    .rename_categories(['Assoc', 'Assis', 'Full', 'Endow', 'Disting']))
salary_df.info()

# +
fig, axes = plt.subplots(
    nrows=2, ncols=2,
    figsize=(10, 6),
    sharey=True)

departments = ['BFIN', 'CHEM', 'PSY', 'ENG']
for department, ax in zip(departments, axes.flatten()):
    df = salary_df[salary_df['Org'] == department]
    sns.stripplot(x='Pos', y='Salary', data=df, ax=ax)
    ax.set_title(f'{department}\'s Salary')

fig.tight_layout()
# -

kernel = NUTS(glm_metric.multi_nominal_predictors, target_accept_prob=0.99)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=10000, num_chains=2)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(salary_df['Salary'].values),
    grp=jnp.concatenate([salary_df[c].cat.codes.values[..., None]
                        for c in ['Org', 'Pos']], axis=1),
    nb_groups=(salary_df['Org'].cat.categories.size,
               salary_df['Pos'].cat.categories.size),
)
mcmc.print_summary()

idata_hom = az.from_numpyro(
    mcmc,
    coords=dict(org=salary_df['Org'].cat.categories.values,
                pos=salary_df['Pos'].cat.categories.values),
    dims=dict(b1=['org'],
              b2=['pos'],
              b1b2=['org', 'pos']),
)
az.plot_trace(idata_hom)
plt.tight_layout()

# +
fig, axes = plt.subplots(
    nrows=2, ncols=2,
    figsize=(10, 6),
    sharey=True)

n_curves = 20
for org, ax in zip(departments, axes.flatten()):
    # Plot data.
    df = salary_df[salary_df['Org'] == org]
    sns.stripplot(x='Pos', y='Salary', data=df, ax=ax)
    ax.set_title(f'{org}\'s Salary with Pred. Post. Distrib.')

    # Plot posterior distribution.
    posterior = idata_hom.posterior
    b0 = posterior['b0'].values.flatten()
    b1 = posterior['b1'].sel(org=org).values.flatten()
    y_sigma = posterior['y_sigma'].values.flatten()

    curve_indices = np.random.choice(
        posterior.draw.size * posterior.chain.size, n_curves, replace=False)

    for pos_i, pos in enumerate(salary_df['Pos'].cat.categories):
        b2 = posterior['b2'].sel(pos=pos).values.flatten()
        b1b2 = posterior['b1b2'].sel(org=org, pos=pos).values.flatten()

        mean = b0 + b1 + b2 + b1b2

        for idx in curve_indices:
            rv = norm(mean[idx], y_sigma[idx])
            yrange = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 1000)
            xpdf = rv.pdf(yrange)

            # Scale xpdf
            xpdf = xpdf * 0.75 / np.max(xpdf)

            # Plot pdf curve.
            ax.plot(pos_i - xpdf, yrange, c='b', alpha=.1)

fig.tight_layout()
# -

# ### Contrasts


def plot_contrasts(
        idata, coef: str,
        left_sel: 'dict[str, list[str]]',
        right_sel: 'dict[str, list[str]]',
        comp_val: float = None,
        rope: 'tuple[float, float]' = None,
        ax: plt.Axes = None):
    """
    Plot contrasts based on inference data from 2 nominal predictors
    and homogeneous normal model.

    Parameters
    ----------
    idata: az.InferenceData
        Inference Data converted from numpyro's MCMC object.
    coef: str
        Name of the coefficient to be used for contrasts comparison.
    left_sel: dict[str, list[str]]
        Selector that will be passed to .sel() function.
    right_sel: dict[str, list[str]]
        Selector that will be passed to .sel() function.
    ax: plt.Axes
        Axes to be used. If None, then new axes will be created.
    """
    def average_last_dim_if_necessary(values):
        if values.ndim == 2:
            return values

        values = np.reshape(values, (*values.shape[:2], -1))
        return np.mean(values, axis=-1)

    if ax is None:
        _, ax = plt.subplots()

    posterior = idata.posterior
    left_values = average_last_dim_if_necessary(
        posterior[coef].sel(left_sel).values)
    right_values = average_last_dim_if_necessary(
        posterior[coef].sel(right_sel).values)

    diff = left_values - right_values
    az.plot_posterior(
        diff, point_estimate='median', hdi_prob=0.95, ref_val=comp_val, rope=rope, ax=ax)
    ax.set_xlabel('Difference')

    return ax


# #### Main Effect Contrasts

# +
pos_contrasts = [
    dict(left='Full', right='Assoc', comp_val=0.0, rope=(-1000, 1000)),
    dict(left='Assoc', right='Assis', comp_val=0.0, rope=(-1000, 1000)),
]

fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
for contrast, ax in zip(pos_contrasts, axes):
    plot_contrasts(
        idata_hom, 'b2',
        left_sel=dict(pos=contrast['left']),
        right_sel=dict(pos=contrast['right']),
        comp_val=contrast['comp_val'],
        rope=contrast['rope'],
        ax=ax)
    ax.set_title(f'{contrast["left"]} vs {contrast["right"]}')

fig.tight_layout()

# +
org_contrasts = [
    dict(left='CHEM', right='ENG', comp_val=0.0, rope=(-1000, 1000)),
    dict(left='CHEM', right='PSY', comp_val=0.0, rope=(-1000, 1000)),
    dict(left='BFIN', right=['PSY', 'CHEM', 'ENG'],
         comp_val=0.0, rope=(-1000, 1000)),
]

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
for contrast, ax in zip(org_contrasts, axes):
    plot_contrasts(
        idata_hom, 'b1',
        left_sel=dict(org=contrast['left']),
        right_sel=dict(org=contrast['right']),
        comp_val=contrast['comp_val'],
        rope=contrast['rope'],
        ax=ax)
    ax.set_title(f'{contrast["left"]} vs {contrast["right"]}')

fig.tight_layout()
# -

# #### Interaction Contrasts
# TODO

# ## Heterogeneous Variances and Robustness against Outliers

kernel = NUTS(
    glm_metric.multi_nominal_predictors_het_var_robust,
    init_strategy=init_to_median)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=5000, num_chains=2)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(salary_df['Salary'].values),
    grp=jnp.concatenate([salary_df[c].cat.codes.values[..., None]
                        for c in ['Org', 'Pos']], axis=1),
    y_sds=jnp.array(salary_df.groupby(['Org', 'Pos']).std().dropna().values),
    nb_groups=(salary_df['Org'].cat.categories.size,
               salary_df['Pos'].cat.categories.size),
)
mcmc.print_summary()

idata_het = az.from_numpyro(
    mcmc,
    coords=dict(org=salary_df['Org'].cat.categories.values,
                pos=salary_df['Pos'].cat.categories.values),
    dims=dict(b1=['org'],
              b2=['pos'],
              b1b2=['org', 'pos'],
              y_sigma=['org', 'pos']),
)
az.plot_trace(idata_het, ['b1', 'b2', 'b1b2', 'y_sigma'])
plt.tight_layout()

# +
fig, axes = plt.subplots(
    nrows=2, ncols=2,
    figsize=(10, 6),
    sharey=True)

n_curves = 20
for org, ax in zip(departments, axes.flatten()):
    # Plot data.
    df = salary_df[salary_df['Org'] == org]
    sns.stripplot(x='Pos', y='Salary', data=df, ax=ax)
    ax.set_title(f'{org}\'s Salary with Pred. Post. Distrib.')

    # Plot posterior distribution.
    posterior = idata_het.posterior
    b0 = posterior['b0'].values.flatten()
    b1 = posterior['b1'].sel(org=org).values.flatten()
    nu = posterior['nu'].values.flatten()

    curve_indices = np.random.choice(
        posterior.draw.size * posterior.chain.size, n_curves, replace=False)

    for pos_i, pos in enumerate(salary_df['Pos'].cat.categories):
        b2 = posterior['b2'].sel(pos=pos).values.flatten()
        b1b2 = posterior['b1b2'].sel(org=org, pos=pos).values.flatten()
        y_sigma = posterior['y_sigma'].sel(org=org, pos=pos).values.flatten()

        mean = b0 + b1 + b2 + b1b2

        for idx in curve_indices:
            rv = t(nu[idx], mean[idx], y_sigma[idx])
            yrange = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 1000)
            xpdf = rv.pdf(yrange)

            # Scale xpdf
            xpdf = xpdf * 0.75 / np.max(xpdf)

            # Plot pdf curve.
            ax.plot(pos_i - xpdf, yrange, c='b', alpha=.1)

    ax.set_ylim(0, 400000)

fig.tight_layout()
# -

az.plot_posterior(idata_het, ['nu', 'y_sigma'])
plt.tight_layout()

# ## Split-plot Design

agri_df = pd.read_csv('datasets/SplitPlotAgriData.csv')
agri_df['Field'] = agri_df['Field'].astype('category')
agri_df['Till'] = agri_df['Till'].astype('category')
agri_df['Fert'] = agri_df['Fert'].astype('category')
agri_df.info()

# +
fig, axes = plt.subplots(ncols=3, sharey=True)

for till, ax in zip(agri_df['Till'].cat.categories, axes):
    df = agri_df[agri_df['Till'] == till]

    for field in df['Field'].cat.categories:
        df_ = df[df['Field'] == field]
        sns.lineplot(x='Fert', y='Yield', data=df_, ax=ax)

    ax.set_title(f'{till} Tilling')
    ax.set_xlim(-0.5, 2.5)

fig.tight_layout()
