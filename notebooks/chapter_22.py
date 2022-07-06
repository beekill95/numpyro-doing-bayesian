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
from numpyro.infer.initialization import init_to_feasible, init_to_median
import numpyro_glm
import numpyro_glm.logistic.models as glm_logistic
import pandas as pd
from scipy.special import expit
from scipy.stats import beta
import seaborn as sns

numpyro.set_host_device_count(4)
# -

# # Chapter 22: Nominal Predicted Variable
# ## Softmax Regression

data1_df: pd.DataFrame = pd.read_csv(
    'datasets/SoftmaxRegData1.csv', dtype={'Y': 'category'})
data1_df.info()

sns.scatterplot(x='X1', y='X2', style='Y', hue='Y', data=data1_df)
plt.tight_layout()

kernel = NUTS(glm_logistic.softmax_multi_metric_predictors,
              init_strategy=init_to_median)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(data1_df['Y'].cat.codes.values),
    x=jnp.array(data1_df[['X1', 'X2']].values),
    K=data1_df['Y'].cat.categories.size,
)
mcmc.print_summary()

idata = az.from_numpyro(
    mcmc,
    coords=dict(group=data1_df['Y'].cat.categories,
                pred=['X1', 'X2']),
    dims=dict(b0=['group'], b=['group', 'pred']),
)
az.plot_trace(idata, ['b0', 'b'])
plt.tight_layout()

# +
fig, axes = plt.subplots(
    nrows=data1_df['Y'].cat.categories.size, ncols=3, figsize=(12, 16))

posterior = idata.posterior
for i, group in enumerate(data1_df['Y'].cat.categories):
    if i == 0:
        # TODO: right now, plotting the first (ref) category,
        # will make the plot super ugly.
        continue

    for j, coeff in enumerate(['b0', 'X1', 'X2']):
        ax = axes[i, j]
        vals = (posterior['b0'].sel(group=group) if coeff == 'b0'
                else posterior['b'].sel(group=group, pred=coeff)).values.flatten()

        az.plot_posterior(vals, kind='hist',
                          point_estimate='mode', hdi_prob=0.95, ax=ax)
        ax.set_title(f'Out: {group}. Pred: {coeff}')

fig.tight_layout()
# -


# ## Conditional Logistic Model

cond1_df = pd.read_csv(
    'datasets/CondLogistRegData1.csv', dtype={'Y': 'category'})
cond1_df.info()

# ### Model 1
# ![](figures/c22_conditional_model_1.png)

# +
from jax.scipy.special import expit  # noqa
import numpyro.distributions as dist  # noqa


def conditional_model_1(y: jnp.ndarray, x: jnp.ndarray, K: int):
    assert y.shape[0] == x.shape[0]
    assert K == 4, 'This only works with 4 nominal outcomes.'
    assert x.shape[1] == 2, 'This only works with 2 metric predictors.'

    nb_obs = y.shape[0]
    nb_preds = x.shape[1]

    # Metric predictors statistics.
    x_mean = jnp.mean(x, axis=0)
    x_sd = jnp.std(x, axis=0)

    # Normalize x.
    xz = (x - x_mean) / x_sd

    a0 = numpyro.sample('_a0', dist.Normal(0, 2).expand([K - 1]))
    a = numpyro.sample('_a', dist.Normal(0, 2).expand([K - 1, nb_preds]))

    phi = expit(a0[None, ...] + xz @ a.T)

    # This is the part where we actually calculate the probability of each nominal outcome.
    # Probability of getting the first outcome is phi[:, 0]
    mu0 = phi[:, 0]

    # Probability of getting the second outcome is: (1 - phi[:, 0]) * phi[:, 1],
    # which essentially means it first not belongs to the first outcome,
    # and has to belong to the second outcome.
    mu1 = phi[:, 1] * (1 - phi[:, 0])

    # Similarly, the probability of the third outcome is
    # (1 - phi[:, 0]) * (1 - phi[:, 1]) * phi[:, 2]
    mu2 = phi[:, 2] * (1 - phi[:, 1]) * (1 - phi[:, 0])

    # And the last outcome is:
    mu3 = (1 - phi[:, 2]) * (1 - phi[:, 1]) * (1 - phi[:, 0])
    mu = jnp.c_[mu0, mu1, mu2, mu3]

    with numpyro.plate('obs', nb_obs) as idx:
        numpyro.sample('y', dist.Categorical(mu[idx]), obs=y[idx])

    # Transform to original scale.
    numpyro.deterministic('b0', a0 - jnp.dot(a, x_mean / x_sd))
    numpyro.deterministic('b', a / x_sd)


kernel = NUTS(conditional_model_1,
              init_strategy=init_to_median)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(cond1_df['Y'].cat.codes.values),
    x=jnp.array(cond1_df[['X1', 'X2']].values),
    K=cond1_df['Y'].cat.categories.size,
)
mcmc.print_summary()
# -

idata = az.from_numpyro(
    mcmc,
    coords=dict(level=[1, 2, 3], pred=['X1', 'X2']),
    dims=dict(b=['level', 'pred'], b0=['level']))
az.plot_trace(idata, ['b', 'b0'])
plt.tight_layout()

# +
from scipy.special import expit  # noqa

fig: plt.Figure = plt.figure(figsize=(15, 6))
gs = fig.add_gridspec(nrows=3, ncols=5)
posterior = idata.posterior

# Plot data scatter with superimposed 0.5 prob lines.
ax = fig.add_subplot(gs[:, :2])
sns.scatterplot(x='X1', y='X2', style='Y', hue='Y', data=cond1_df, ax=ax)
xx, yy = np.meshgrid(
    np.linspace(*ax.get_xlim(), 1000),
    np.linspace(*ax.get_ylim(), 1000),
    indexing='ij',
)
n_lines = 20
for level in [1, 2, 3]:
    b0 = posterior['b0'].sel(level=level).values.flatten()
    b1 = posterior['b'].sel(level=level, pred='X1').values.flatten()
    b2 = posterior['b'].sel(level=level, pred='X2').values.flatten()

    indices = np.random.choice(
        posterior.draw.size * posterior.chain.size, n_lines, replace=False)
    for idx in indices:
        p = expit(b0[idx] + b1[idx] * xx + b2[idx] * yy)
        ax.contour(xx, yy, p, colors='blue', alpha=.2, levels=[.5])

for i, level in enumerate([1, 2, 3]):
    for j, coeff in enumerate(['b0', 'X1', 'X2']):
        ax = fig.add_subplot(gs[i, j + 2])
        vals = (posterior['b0'].sel(level=level) if coeff == 'b0'
                else posterior['b'].sel(level=level, pred=coeff)).values.flatten()

        az.plot_posterior(vals, kind='hist',
                          point_estimate='mode', hdi_prob=0.95, ax=ax)
        ax.set_title(f'Lambda: {level}. Pred: {coeff}')

fig.tight_layout()
