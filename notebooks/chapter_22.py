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

idata = az.from_numpyro(mcmc)
az.plot_trace(idata, ['b0', 'b'])
plt.tight_layout()

# ## Conditional Logistic Model
