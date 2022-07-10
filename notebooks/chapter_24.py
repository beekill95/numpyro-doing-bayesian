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
import numpyro_glm.count.models as glm_count
import numpyro_glm.ordinal.plots as ordinal_plots
import pandas as pd
import seaborn as sns

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
    dims=dict(
        Hair=hair_eye_df['Hair'].cat.categories,
        Eye=hair_eye_df['Eye'].cat.categories),
    coords=dict(b1=['Eye'], b2=['Hair'], b1b2=['Eye', 'Hair']))
az.plot_trace(idata, ['b1', 'b2', 'b1b2'])
plt.tight_layout()
