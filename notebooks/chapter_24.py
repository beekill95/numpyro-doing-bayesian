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
#   title: '[Doing Bayesian Data Analysis] Chapter 24: Count Predicted Variable'
# ---

# %cd ..
# %load_ext autoreload
# %autoreload 2

# +
import arviz as az
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro_glm.count.models as glm_count
import pandas as pd

numpyro.set_host_device_count(4)
# -

# # Chapter 24: Count Predicted Variable
# ## Poisson Exponential Model
# ### Example: Hair Eye Go Again

hair_eye_df: pd.DataFrame = pd.read_csv(
    'datasets/HairEyeColor.csv',
    dtype=dict(Hair='category', Eye='category'))
hair_eye_df.info()

kernel = NUTS(glm_count.two_nominal_predictors)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(hair_eye_df['Count'].values),
    x1=jnp.array(hair_eye_df['Eye'].cat.codes.values),
    x2=jnp.array(hair_eye_df['Hair'].cat.codes.values),
    K1=hair_eye_df['Eye'].cat.categories.size,
    K2=hair_eye_df['Hair'].cat.categories.size,
)
mcmc.print_summary()

idata = az.from_numpyro(
    mcmc,
    coords=dict(
        Hair=hair_eye_df['Hair'].cat.categories,
        Eye=hair_eye_df['Eye'].cat.categories),
    dims=dict(
        b1=['Eye'], b2=['Hair'],
        b1b2=['Eye', 'Hair'],
        P=['Eye', 'Hair'],
        P_x1=['Eye'], P_x2=['Hair'])
)
az.plot_trace(idata, ['b1', 'b2', 'b1b2'])
plt.tight_layout()

# +
fig, axes = plt.subplots(
    nrows=hair_eye_df['Eye'].cat.categories.size,
    ncols=hair_eye_df['Hair'].cat.categories.size,
    figsize=(16, 16))

posterior = idata['posterior']
for i, eye in enumerate(hair_eye_df['Eye'].cat.categories):
    for j, hair in enumerate(hair_eye_df['Hair'].cat.categories):
        ax = axes[i, j]
        p = posterior['P'].sel(Eye=eye, Hair=hair).values
        az.plot_posterior(p, hdi_prob=.95, point_estimate='mode', ax=ax)
        ax.set_title(f'Eye: {eye} - Hair: {hair}')

fig.tight_layout()

# +
fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

blue_eye_black_hair = posterior['b1b2'].sel(Eye='Blue', Hair='Black').values
brown_eye_black_hair = posterior['b1b2'].sel(Eye='Brown', Hair='Black').values

blue_eye_blond_hair = posterior['b1b2'].sel(Eye='Blue', Hair='Blond').values
brown_eye_blond_hair = posterior['b1b2'].sel(Eye='Brown', Hair='Blond').values

diff_black = blue_eye_black_hair - brown_eye_black_hair
diff_blond = blue_eye_blond_hair - brown_eye_blond_hair
diff = diff_black - diff_blond

ax = axes[0]
az.plot_posterior(
    diff_black, hdi_prob=.95, point_estimate='mode', ref_val=0, rope=(-0.1, 0.1), ax=ax)
ax.set_title('Blue - Brown @ Black')
ax.set_xlabel('Beta Deflect. Diff.')

ax = axes[1]
az.plot_posterior(
    diff_blond, hdi_prob=.95, point_estimate='mode', ref_val=0, rope=(-0.1, 0.1), ax=ax)
ax.set_title('Blue - Brown @ Blond')
ax.set_xlabel('Beta Deflect. Diff.')

ax = axes[2]
az.plot_posterior(
    diff, hdi_prob=.95, point_estimate='mode', ref_val=0, rope=(-0.1, 0.1), ax=ax)
ax.set_title('Blue.v.Brown\n(x)\nBlack.v.Blond')
ax.set_xlabel('Beta Deflect. Diff. of Diff.')

fig.tight_layout()
