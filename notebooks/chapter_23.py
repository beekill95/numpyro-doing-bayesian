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
              target_accept_prob=0.99)
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
