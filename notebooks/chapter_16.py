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
#     display_name: doing_bayes
#     language: python
#     name: doing_bayes
# ---

# %cd ..
# %load_ext autoreload
# %autoreload 2

import arviz as az
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro_glm
import numpyro_glm.metric.models as glm_metric
import numpyro_glm.metric.plots as plots
import pandas as pd
from scipy.stats import norm, t

# # Chapter 16: Metric-Predicted Variable on One or Two Groups

# ## Estimating Mean and Standard Deviation of a Normal distribution

# ### Metric Model

numpyro.render_model(
    glm_metric.one_group,
    model_args=(jnp.ones(5), ),
    render_params=True)

# ### Synthesis data

# +
MEAN = 5
STD = 3
N = 100

y = np.random.normal(MEAN, STD, N)

# Plot the histogram and true PDF of normal distribution.
fig, ax = plt.subplots()
ax.hist(y, density=True, label='Histogram of $y$')
xmin, xmax = ax.get_xlim()
p = np.linspace(xmin, xmax, 1000)
ax.plot(p, norm.pdf(p, loc=MEAN, scale=STD), label='Normal PDF')
ax.legend()
fig.tight_layout()
# -

# Now, we'll try to apply the metric model on that data to see
# if it can recover the parameter.

mcmc_key = random.PRNGKey(0)
kernel = NUTS(glm_metric.one_group)
mcmc = MCMC(kernel, num_warmup=250, num_samples=750)
mcmc.run(mcmc_key, jnp.array(y))
mcmc.print_summary()

# Plot diagnostics plot to see if the MCMC chains are well-behaved.

numpyro_glm.plot_diagnostic(mcmc, ['mean', 'std'])

fig = plots.plot_st(mcmc, y)

# ### Smart Group IQ

iq_data = pd.read_csv('datasets/TwoGroupIQ.csv')
iq_data['Group'] = iq_data['Group'].astype('category')
smart_group_data = iq_data[iq_data.Group == 'Smart Drug']
smart_group_data.describe()

# Then, we will apply the one group model to the data
# and plot the results.

mcmc_key = random.PRNGKey(0)
kernel = NUTS(glm_metric.one_group)
mcmc = MCMC(kernel, num_warmup=250, num_samples=750)
mcmc.run(mcmc_key, jnp.array(smart_group_data.Score.values))
mcmc.print_summary()

numpyro_glm.plot_diagnostic(mcmc, ['mean', 'std'])

fig = plots.plot_st(
    mcmc, smart_group_data.Score.values,
    mean_comp_val=100,
    std_comp_val=15,
    effsize_comp_val=0,
)

# + [markdown] tags=[]
# ## Outliers and Robust Estimation: $t$ Distribution
# -

# ### Robust Metric Model

numpyro.render_model(
    glm_metric.one_group_robust,
    model_args=(jnp.ones(5),),
    render_params=True)

# ### Synthesis Data

# +
MEAN = 5
SIGMA = 3
NORMALITY = 3
N = 1000

y = np.random.standard_t(NORMALITY, size=N) * SIGMA + MEAN

# Plot the histogram and true PDF of normal distribution.
fig, ax = plt.subplots()
ax.hist(y, density=True, bins=100, label='Histogram of $y$')
xmin, xmax = ax.get_xlim()
p = np.linspace(xmin, xmax, 1000)
ax.plot(p, t.pdf(p, loc=MEAN, scale=SIGMA, df=NORMALITY), label='Student-$t$ PDF')
ax.plot(p, norm.pdf(p, loc=y.mean(), scale=y.std()),
        label='Normal PDF using\ndata mean and std')
ax.legend()
fig.tight_layout()
# -

# Using the robust metric model on our synthesis data
# to see if it can recover the original parameters.

mcmc_key = random.PRNGKey(0)
kernel = NUTS(glm_metric.one_group_robust)
mcmc = MCMC(kernel, num_warmup=250, num_samples=750)
mcmc.run(mcmc_key, jnp.array(y))
mcmc.print_summary()

numpyro_glm.plot_diagnostic(mcmc, ['mean', 'sigma', 'nu'])

fig = plots.plot_st_2(mcmc, y)

# ### Smart Drug Group Data

mcmc_key = random.PRNGKey(0)
kernel = NUTS(glm_metric.one_group_robust)
mcmc = MCMC(kernel, num_warmup=250, num_samples=750)
mcmc.run(mcmc_key, jnp.array(smart_group_data.Score.values))
mcmc.print_summary()

numpyro_glm.plot_diagnostic(mcmc, ['mean', 'sigma', 'nu'])

fig = plots.plot_st_2(
    mcmc, smart_group_data.Score.values,
    mean_comp_val=100,
    sigma_comp_val=15,
    effsize_comp_val=0)

fig = numpyro_glm.plot_pairwise_scatter(mcmc, ['mean', 'sigma', 'nu'])

# ## Two Groups

mcmc_key = random.PRNGKey(0)
kernel = NUTS(glm_metric.multi_groups_robust)
mcmc = MCMC(kernel, num_warmup=250, num_samples=750)
mcmc.run(
    mcmc_key,
    jnp.array(iq_data['Score'].values),
    jnp.array(iq_data['Group'].cat.codes.values),
    len(iq_data['Group'].cat.categories),
)
mcmc.print_summary()

numpyro_glm.plot_diagnostic(mcmc, ['mean', 'sigma', 'nu'])

# Plot the resulting posteriors.

fig = plots.plot_st_2(
    mcmc,
    iq_data[iq_data['Group'] == 'Placebo']['Score'].values,
    mean_coords=dict(mean_dim_0=0),
    sigma_coords=dict(sigma_dim_0=0),
    figtitle='Placebo Posteriors')

fig = plots.plot_st_2(
    mcmc,
    iq_data[iq_data['Group'] == 'Smart Drug']['Score'].values,
    mean_coords=dict(mean_dim_0=1),
    sigma_coords=dict(sigma_dim_0=1),
    figtitle='Smart Drug Posteriors')

# Then, we will plot the difference between the two groups.

# +
idata = az.from_numpyro(mcmc)
posteriors = idata.posterior

fig, axes = plt.subplots(ncols=3, figsize=(18, 6))
fig.suptitle('Difference between Smart Drug and Placebo')

# Plot mean difference.
ax = axes[0]
mean_difference = (posteriors['mean'].sel(dict(mean_dim_0=1))
                   - posteriors['mean'].sel(dict(mean_dim_0=0)))
az.plot_posterior(
    mean_difference,
    hdi_prob=0.95,
    ref_val=0,
    rope=(-0.5, 0.5),
    point_estimate='mode',
    kind='hist',
    ax=ax)
ax.set_title('Difference of Means')
ax.set_xlabel('$\mu[1] - \mu[0]$')

# Plot sigma difference.
ax = axes[1]
sigma_difference = (posteriors['sigma'].sel(dict(sigma_dim_0=1))
                    - posteriors['sigma'].sel(dict(sigma_dim_0=0)))
az.plot_posterior(
    sigma_difference,
    hdi_prob=0.95,
    ref_val=0,
    rope=(-0.5, 0.5),
    point_estimate='mode',
    kind='hist',
    ax=ax)
ax.set_title('Difference of Scales')
ax.set_xlabel('$\sigma[1] - \sigma[0]$')

# Plot effect size.
ax = axes[2]
sigmas_squared = (posteriors['sigma'].sel(dict(sigma_dim_0=1))**2
                  + posteriors['sigma'].sel(dict(sigma_dim_0=0))**2)
effect_size = mean_difference / np.sqrt(sigmas_squared)
az.plot_posterior(
    effect_size,
    hdi_prob=0.95,
    ref_val=0,
    rope=(-0.1, 0.1),
    point_estimate='mode',
    kind='hist',
    ax=ax)
ax.set_title('Effect Size')
ax.set_xlabel('$(\mu[1] - \mu[0]) / \sqrt{\sigma[1]^2 + \sigma[0]^2}$')

fig.tight_layout()
