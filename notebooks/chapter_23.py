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
import numpyro_glm.ordinal.models as glm_ordinal
import numpyro_glm.ordinal.plots as ordinal_plots
import pandas as pd
import seaborn as sns

numpyro.set_host_device_count(4)
# -

# # Chapter 23: Ordinal Predicted Variable
#
# ## The Case of A Single Group

ord_1_df = pd.read_csv('datasets/OrdinalProbitData-1grp-1.csv')
yord_1_cat = pd.CategoricalDtype([1, 2, 3, 4, 5, 6, 7], ordered=True)
ord_1_df['Y'] = ord_1_df['Y'].astype(yord_1_cat)

kernel = NUTS(glm_ordinal.yord_single_group,
              init_strategy=init_to_median,
              target_accept_prob=0.999)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(ord_1_df['Y'].cat.codes.values),
    K=yord_1_cat.categories.size,
)
mcmc.print_summary()

idata_yord_1 = az.from_numpyro(mcmc)
az.plot_trace(idata_yord_1)
plt.tight_layout()


# +
def plot_ordinal_one_group_results(idata, df, thresholds):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))

    # Plot latent score's mean posterior distribution.
    ax = axes[0, 0]
    az.plot_posterior(
        idata,
        var_names='mu',
        hdi_prob=0.95,
        ref_val=2,
        point_estimate='mode',
        ax=ax)
    ax.set_title('Mean')
    ax.set_xlabel('$\mu$')

    # Plot data with posterior distribution.
    ax = axes[0, 1]
    ordinal_plots.plot_ordinal_data_with_posterior(
        idata, 'mu', 'sigma', thresholds,
        df, ordinal_predicted='Y',
        ax=ax)
    ax.set_title('Data with Post. Pred. Distrib.')

    # Plot latent score's standard deviation.
    ax = axes[1, 0]
    az.plot_posterior(
        idata,
        var_names='sigma',
        hdi_prob=0.95,
        point_estimate='mode',
        ax=ax)
    ax.set_title('Std.')
    ax.set_xlabel('$\sigma$')

    # Plot effect size.
    ax = axes[1, 1]
    posterior = idata.posterior
    eff_size = (posterior['mu'] - 2) / posterior['sigma']
    az.plot_posterior(
        eff_size,
        hdi_prob=0.95,
        point_estimate='mode',
        ref_val=0,
        ax=ax)
    ax.set_title('Effect Size')
    ax.set_xlabel('$(\mu - 2) / \sigma$')

    # Plot thresholds.
    ax = axes[2, 0]
    ordinal_plots.plot_threshold_scatter(idata, thresholds, ax=ax)

    axes[2, 1].remove()
    fig.tight_layout()


thresholds = ['thres_1', 'thres_2', 'thres_3', 'thres_4', 'thres_5', 'thres_6']
plot_ordinal_one_group_results(idata_yord_1, ord_1_df, thresholds)
# -

# ---
#
# The same model but applied on different dataset.

ord_2_df = pd.read_csv('datasets/OrdinalProbitData-1grp-2.csv')
yord_2_cat = pd.CategoricalDtype([1, 2, 3, 4, 5, 6, 7], ordered=True)
ord_2_df['Y'] = ord_2_df['Y'].astype(yord_2_cat)

kernel = NUTS(glm_ordinal.yord_single_group,
              init_strategy=init_to_median,
              target_accept_prob=0.99)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(ord_2_df['Y'].cat.codes.values),
    K=yord_2_cat.categories.size,
)
mcmc.print_summary()

idata_yord_2 = az.from_numpyro(mcmc)
az.plot_trace(idata_yord_2)
plt.tight_layout()

plot_ordinal_one_group_results(idata_yord_2, ord_2_df, thresholds)

# ## The Case of Two Groups

two_groups_df = pd.read_csv('datasets/OrdinalProbitData1.csv')
two_groups_ordinal_cat = pd.CategoricalDtype([1, 2, 3, 4, 5], ordered=True)
two_groups_df['Y'] = two_groups_df['Y'].astype(two_groups_ordinal_cat)
two_groups_df['X'] = two_groups_df['X'].astype('category')
two_groups_df.info()

kernel = NUTS(glm_ordinal.yord_two_groups,
              init_strategy=init_to_median,
              target_accept_prob=0.99)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(two_groups_df['Y'].cat.codes.values),
    grp=jnp.array(two_groups_df['X'].cat.codes.values),
    K=two_groups_ordinal_cat.categories.size,
    nb_groups=two_groups_df['X'].cat.categories.size,
)
mcmc.print_summary()

idata_two_groups = az.from_numpyro(
    mcmc,
    coords=dict(group=two_groups_df['X'].cat.categories),
    dims=dict(sigma=['group'], mu=['group'])
)
az.plot_trace(idata_two_groups)
plt.tight_layout()

# +
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 16))

# Plot group A's latent mean.
ax = axes[0, 0]
az.plot_posterior(
    idata_two_groups,
    var_names='mu',
    coords=dict(group='A'),
    point_estimate='mode',
    hdi_prob=0.95,
    ax=ax)
ax.set_title('A Mean')
ax.set_xlabel('$\mu_A$')

# Plot group A's data with posterior distrib.
ax = axes[0, 1]
ordinal_plots.plot_ordinal_data_with_posterior(
    idata_two_groups,
    'mu', 'sigma',
    ['thres_1', 'thres_2', 'thres_3', 'thres_4'],
    latent_coords=dict(group='A'),
    data=two_groups_df[two_groups_df['X'] == 'A'],
    ordinal_predicted='Y',
    ax=ax,
)
ax.set_title('Data for A with Post. Pred.')

# Plot group B's latent mean.
ax = axes[1, 0]
az.plot_posterior(
    idata_two_groups,
    var_names='mu',
    coords=dict(group='B'),
    point_estimate='mode',
    hdi_prob=0.95,
    ax=ax)
ax.set_title('B Mean')
ax.set_xlabel('$\mu_B$')

# Plot group B's data with posterior distrib.
ax = axes[1, 1]
thresholds = ['thres_1', 'thres_2', 'thres_3', 'thres_4']
ordinal_plots.plot_ordinal_data_with_posterior(
    idata_two_groups,
    'mu', 'sigma',
    thresholds,
    latent_coords=dict(group='B'),
    data=two_groups_df[two_groups_df['X'] == 'B'],
    ordinal_predicted='Y',
    ax=ax,
)
ax.set_title('Data for B with Post. Pred.')

# Plot A's standard deviation.
ax = axes[2, 0]
az.plot_posterior(
    idata_two_groups,
    var_names='sigma',
    coords=dict(group='A'),
    point_estimate='mode',
    hdi_prob=0.95,
    ax=ax)
ax.set_title('A Std.')
ax.set_xlabel('$\sigma_A$')

# Plot difference of means.
posterior = idata_two_groups['posterior']
ax = axes[2, 1]
mean_diff = (posterior['mu'].sel(group='B')
             - posterior['mu'].sel(group='A'))
az.plot_posterior(
    mean_diff,
    point_estimate='mode',
    hdi_prob=0.95,
    ax=ax)
ax.set_title('Difference of Means')
ax.set_xlabel('$\mu_B - \mu_A$')

# Plot B's standard deviation.
ax = axes[3, 0]
az.plot_posterior(
    idata_two_groups,
    var_names='sigma',
    coords=dict(group='B'),
    point_estimate='mode',
    hdi_prob=0.95,
    ax=ax)
ax.set_title('B Std.')
ax.set_xlabel('$\sigma_B$')

# Plot difference of sigmas.
ax = axes[3, 1]
sigma_diff = (posterior['sigma'].sel(group='B')
              - posterior['sigma'].sel(group='A'))
az.plot_posterior(
    sigma_diff,
    point_estimate='mode',
    hdi_prob=0.95,
    ax=ax)
ax.set_title('Difference of Sigmas')
ax.set_xlabel('$\sigma_B - \sigma_A$')

# Plot threshold scatter.
ax = axes[4, 0]
ordinal_plots.plot_threshold_scatter(
    idata_two_groups,
    thresholds,
    ax=ax
)
ax.set_xlabel('Threshold')
ax.set_ylabel('Mean Threshold')

# Plot effect size.
ax = axes[4, 1]
sigma_squared = (posterior['sigma'].sel(group='A')**2
                 + posterior['sigma'].sel(group='B')**2)
eff_size = mean_diff / np.sqrt(sigma_squared / 2)
az.plot_posterior(
    eff_size,
    point_estimate='mode',
    hdi_prob=0.95,
    ax=ax)
ax.set_title('Effect Size')
ax.set_xlabel('$(\mu_B - \mu_A) / \sqrt{\sigma_A^2 + \sigma_B^2}$')

fig.tight_layout()
