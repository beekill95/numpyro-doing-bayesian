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
from functools import reduce
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro_glm.metric.models as glm_metric
import pandas as pd
import seaborn as sns
from scipy.stats import norm, t

numpyro.set_host_device_count(4)
# -

# # Chapter 19: Metric Predicted Variable with One Nominal Predictor
# ## Hierarchical Bayesian Approach

fruit_df = pd.read_csv('datasets/FruitflyDataReduced.csv')
fruit_df['CompanionNumber'] = fruit_df['CompanionNumber'].astype('category')
fruit_df.info()

fruit_df.describe()

# Run the model.

key = random.PRNGKey(0)
kernel = NUTS(glm_metric.one_nominal_predictor)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    key,
    jnp.array(fruit_df['Longevity'].values),
    jnp.array(fruit_df['CompanionNumber'].cat.codes.values),
    len(fruit_df['CompanionNumber'].cat.categories))
mcmc.print_summary()

idata = az.from_numpyro(
    mcmc,
    coords=dict(CompanionNumber=fruit_df['CompanionNumber'].cat.categories),
    dims=dict(b_grp=['CompanionNumber']))
az.plot_trace(idata)
plt.tight_layout()

# Plot the posterior results.

# +
fig, ax = plt.subplots()
sns.swarmplot(fruit_df['CompanionNumber'], fruit_df['Longevity'], ax=ax)
ax.set_xlim(xmin=-1)

b0 = idata.posterior['b0'].values.flatten()
b_grp = {name: idata.posterior['b_grp'].sel(CompanionNumber=name).values.flatten()
         for name in fruit_df['CompanionNumber'].cat.categories}
ySigma = idata.posterior['y_sigma'].values.flatten()

n_curves = 20
for i, name in enumerate(fruit_df['CompanionNumber'].cat.categories):
    indices = np.random.choice(
        len(idata.posterior.draw) * len(idata.posterior.chain), n_curves, replace=False)

    for idx in indices:
        rv = norm(b0[idx] + b_grp[name][idx], ySigma[idx])
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
        left = mean([posterior['b_grp'].sel(CompanionNumber=n).values
                     for n in contrast['left']])
        right = mean([posterior['b_grp'].sel(CompanionNumber=n).values
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
        effSize = diff / posterior['y_sigma']

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
# -

# ## Including a Metric Predictor

kernel = NUTS(glm_metric.one_nominal_one_metric)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(fruit_df['Longevity'].values),
    grp=jnp.array(fruit_df['CompanionNumber'].cat.codes.values),
    cov=jnp.array(fruit_df['Thorax'].values),
    nb_groups=len(fruit_df['CompanionNumber'].cat.categories),
)
mcmc.print_summary()

idata_met = az.from_numpyro(
    mcmc,
    coords=dict(CompanionNumber=fruit_df['CompanionNumber'].cat.categories),
    dims=dict(b_grp=['CompanionNumber']))
az.plot_trace(idata_met)
plt.tight_layout()

# +
fig, axes = plt.subplots(
    ncols=fruit_df['CompanionNumber'].cat.categories.size,
    figsize=(15, 6),
    sharey=True)

posterior_met = idata_met.posterior
b0 = posterior_met['b0'].values.flatten()
b_cov = posterior_met['b_cov'].values.flatten()
y_sigma = posterior_met['y_sigma'].values.flatten()

n_lines = 20
for companion_nb, ax in zip(fruit_df['CompanionNumber'].cat.categories, axes.flatten()):
    data = fruit_df[fruit_df['CompanionNumber'] == companion_nb]
    sns.scatterplot(x='Thorax', y='Longevity', data=data, ax=ax)
    ax.set_title(f'{companion_nb} Data\nw. Pred. Post. Dist.')

    xrange = np.linspace(*ax.get_xlim(), 1000)

    line_indices = np.random.choice(
        posterior_met.draw.size * posterior_met.chain.size,
        n_lines,
        replace=False)

    b_grp = posterior_met['b_grp'].sel(
        CompanionNumber=companion_nb).values.flatten()

    for idx in line_indices:
        y = b0[idx] + b_grp[idx] + xrange * b_cov[idx]
        ax.plot(xrange, y, c='b', alpha=.2)

        # Plot predicted posterior distribution of Longevity|Thorax.
        for xidx in [300, 600, 900]:
            rv = norm(y[xidx], y_sigma[idx])
            yrange = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 1000)
            xpdf = rv.pdf(yrange)

            # Scale the pdf.
            xpdf = xpdf * 0.05 / np.max(xpdf)

            # Plot the distribution.
            ax.plot(xrange[xidx] - xpdf, yrange, c='b', alpha=.2)

fig.tight_layout()
# -

_ = plot_contrasts(idata_met, contrasts, figsize=(15, 6))

# ## Heterogeneous Variances and Robustness against Outliers

nonhomo_df = pd.read_csv('datasets/NonhomogVarData.csv')
nonhomo_df['Group'] = nonhomo_df['Group'].astype('category')
nonhomo_df.info()

sns.scatterplot(x='Group', y='Y', data=nonhomo_df)

# ### Homogeneous Model

kernel = NUTS(glm_metric.one_nominal_predictor)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(nonhomo_df['Y'].values),
    grp=jnp.array(nonhomo_df['Group'].cat.codes.values),
    nb_groups=nonhomo_df['Group'].cat.categories.size,
)
mcmc.print_summary()

# +
idata_hom = az.from_numpyro(
    mcmc,
    coords=dict(Group=nonhomo_df['Group'].cat.categories.values),
    dims=dict(b_grp=['Group']))
posterior_hom = idata_hom.posterior

b0 = posterior_hom['b0'].values.flatten()
y_sigma = posterior_hom['y_sigma'].values.flatten()

fig, ax = plt.subplots()
sns.scatterplot(x='Group', y='Y', data=nonhomo_df, ax=ax)
ax.set_title(f'Homogeneous model Pred. Post. Dist.')

n_curves = 20
for gid, group in enumerate(nonhomo_df['Group'].cat.categories):
    curve_indices = np.random.choice(
        posterior_hom.draw.size * posterior_hom.chain.size, n_curves, replace=False)

    b_grp = posterior_hom['b_grp'].sel(Group=group).values.flatten()

    for idx in curve_indices:
        mean = b0[idx] + b_grp[idx]
        sigma = y_sigma[idx]
        rv = norm(mean, sigma)

        yrange = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 1000)
        xpdf = rv.pdf(yrange)

        # Scale pdf to be superimposed on scatterplot.
        xpdf = xpdf * 0.75 / np.max(xpdf)

        # Plot the resulting posterior dist.
        ax.plot(gid - xpdf, yrange, c='b', alpha=0.1)

fig.tight_layout()
# -

# ### Heterogeneous Model

kernel = NUTS(glm_metric.one_nominal_predictor_het_var_robust)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(nonhomo_df['Y'].values),
    grp=jnp.array(nonhomo_df['Group'].cat.codes.values),
    nb_groups=nonhomo_df['Group'].cat.categories.size,
)
mcmc.print_summary()

# +
idata_het = az.from_numpyro(
    mcmc,
    coords=dict(Group=nonhomo_df['Group'].cat.categories.values),
    dims=dict(b_grp=['Group'], y_sigma=['Group']))
posterior_het = idata_het.posterior

b0 = posterior_het['b0'].values.flatten()
nu = posterior_het['nu'].values.flatten()

fig, ax = plt.subplots()
sns.scatterplot(x='Group', y='Y', data=nonhomo_df, ax=ax)
ax.set_title(f'Heterogeneous model Pred. Post. Dist.')

n_curves = 20
for gid, group in enumerate(nonhomo_df['Group'].cat.categories):
    curve_indices = np.random.choice(
        posterior_het.draw.size * posterior_het.chain.size, n_curves, replace=False)

    b_grp = posterior_het['b_grp'].sel(Group=group).values.flatten()
    y_sigma = posterior_het['y_sigma'].sel(Group=group).values.flatten()

    for idx in curve_indices:
        mean = b0[idx] + b_grp[idx]
        sigma = y_sigma[idx]
        rv = t(nu[idx], mean, sigma)

        yrange = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 1000)
        xpdf = rv.pdf(yrange)

        # Scale pdf to be superimposed on scatterplot.
        xpdf = xpdf * 0.75 / np.max(xpdf)

        # Plot the resulting posterior dist.
        ax.plot(gid - xpdf, yrange, c='b', alpha=0.1)

fig.tight_layout()
