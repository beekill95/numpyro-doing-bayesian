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
from __future__ import annotations

import arviz as az
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro_glm.utils.dist as dist_utils
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
import pandas as pd
import seaborn as sns

numpyro.set_host_device_count(4)
# -

# # Chapter 12: Bayesian Approaches to Testing a Point ("Null") Hypothesis
# ## The Estimation Approach
# ### Region of Practical Equivalence
#
# > A parameter value is declared to be not credible, or rejected,
# if its entire ROPE lies outside of the 95% highest density interval (HDI)
# of the posterior distribution of that parameter.
#
# > A parameter value is declared to be accepted for practical purposes
# if that value's ROPE completely contains the 95% HDI of the posterior
# of that parameter.

# ## The Model Comparison Approach
# ### Are Different Groups Equal or Not?
#
# Data are downloaded from [JWarmenhoven's implementation](https://github.com/JWarmenhoven/DBDA-python/blob/master/Notebooks/data/background_music.csv).

music_df: pd.DataFrame = pd.read_csv(
    'datasets/background_music.csv', dtype=dict(CondOfSubj='category'))
music_df['CondOfSubj'] = (music_df['CondOfSubj']
                          .cat.rename_categories(
                              {'1': 'Das Kruschke', '2': 'Mozart',
                               '3': 'Bach', '4': 'Beethoven'}))
music_df.info()


# +
def words_recall_in_different_music_genres_model_comparison(
        genre: jnp.ndarray, nb_trials: jnp.ndarray, nb_corrects: jnp.ndarray, nb_genres: int):
    assert genre.shape[0] == nb_trials.shape[0] == nb_corrects.shape[0]
    nb_obs = genre.shape[0]

    # We will compare between two models:
    # * Model 0 uses condition-specific omega,
    # * Model 1 uses the same omega for all conditions.
    model_probs = jnp.array([0.5, 0.5])
    model = numpyro.sample('model', dist.Categorical(model_probs))

    aP = 1.
    bP = 1.

    # Model 0's omegas.
    a = jnp.repeat(aP, nb_genres)
    b = jnp.repeat(bP, nb_genres)
    omegas = numpyro.sample('omegas', dist.Beta(a, b))

    # Model 1's omega0.
    a0 = aP
    b0 = bP
    omega0 = numpyro.sample('omega0', dist.Beta(a0, b0).expand([nb_genres]))

    # The omega to be used will depend on the model chosen.
    omega = numpyro.deterministic(
        'omega', jnp.where(model == 0, omegas, omega0))

    # Kappa's prior.
    kappa_minus_two = numpyro.sample(
        '_kappa-2', dist_utils.gammaDistFromModeStd(20, 20).expand([nb_genres]))
    kappa = numpyro.deterministic('kappa', kappa_minus_two + 2)

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        recall = numpyro.sample(
            'recall', dist_utils.beta_dist_from_omega_kappa(omega[genre[idx]],
                                                            kappa[genre[idx]]))
        numpyro.sample(
            'correct', dist.Binomial(nb_trials[idx], recall), obs=nb_corrects[idx])


kernel = DiscreteHMCGibbs(
    NUTS(words_recall_in_different_music_genres_model_comparison))
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    genre=jnp.array(music_df['CondOfSubj'].cat.codes.values),
    nb_trials=jnp.array(music_df['nTrlOfSubj'].values),
    nb_corrects=jnp.array(music_df['nCorrOfSubj'].values),
    nb_genres=music_df['CondOfSubj'].cat.categories.size,
)
mcmc.print_summary()
# -

idata = az.from_numpyro(
    mcmc,
    coords=dict(genre=music_df['CondOfSubj'].cat.categories),
    dims=dict(
        omega0=['genre'], omegas=['genre'], omega=['genre'], kappa=['genre']))
az.plot_trace(idata)
plt.tight_layout()

# +
from itertools import combinations  # noqa

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
posterior = idata['posterior']

for (left, right), ax in zip(combinations(music_df['CondOfSubj'].cat.categories, 2),
                             axes.flatten()):
    left_omega = posterior['omega'].sel(genre=left).values
    right_omega = posterior['omega'].sel(genre=right).values

    diff = left_omega - right_omega
    az.plot_posterior(diff, hdi_prob=.95, ref_val=0.0,
                      point_estimate='mode', ax=ax)
    ax.set_title('')
    ax.set_xlabel(f'$\\omega_{{{left}}} - \\omega_{{{right}}}$')

fig.tight_layout()
# -

model_idx = posterior['model'].values.flatten()
sns.lineplot(x=np.indices(model_idx.shape).flatten(), y=model_idx)
plt.tight_layout()
