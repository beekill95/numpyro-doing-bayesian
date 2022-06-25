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
# ## Softmax Model

# ## Conditional Logistic Model
