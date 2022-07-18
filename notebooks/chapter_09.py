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
import numpyro.distributions as dist
import numpyro_glm.utils.dist as dist_utils
from numpyro.infer import MCMC, NUTS
import pandas as pd

numpyro.set_host_device_count(4)
# -

# # Chapter 9: Hierarchical Models
# ## Multiple Coins from a Single Mint
# ### Example: Therapeutic Touch

therapeutic_df: pd.DataFrame = pd.read_csv(
    'datasets/TherapeuticTouchData.csv', dtype=dict(s='category'))
therapeutic_df.info()


def therapeutic_touch(y: jnp.ndarray, s: jnp.ndarray, nb_subjects: int):
    assert y.shape[0] == s.shape[0]

    nb_obs = y.shape[0]

    # Omega prior.
    omega = numpyro.sample('omega', dist.Beta(1, 1))

    # Kappa prior.
    kappa_minus_two = numpyro.sample(
        '_kappa-2', dist_utils.gammaDistFromModeStd(1, 10))
    kappa = numpyro.deterministic('kappa', kappa_minus_two + 2)

    # Each subject's ability.
    theta = numpyro.sample(
        'theta', dist_utils.beta_dist_from_omega_kappa(omega, kappa).expand([nb_subjects]))

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        numpyro.sample('y', dist.Bernoulli(theta[s[idx]]), obs=y[idx])


kernel = NUTS(therapeutic_touch)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(therapeutic_df['y'].values),
    s=jnp.array(therapeutic_df['s'].cat.codes.values),
    nb_subjects=therapeutic_df['s'].cat.categories.size,
)
mcmc.print_summary()
