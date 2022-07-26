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
#   title: '[Doing Bayesian Data Analysis] Chapter 9: Hierarchical Models'
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
from numpyro.infer import MCMC, NUTS
import pandas as pd
import seaborn as sns

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
        '_kappa-2', dist_utils.gamma_dist_from_mean_std(1, 10))
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

idata = az.from_numpyro(
    mcmc,
    coords=dict(subject=therapeutic_df['s'].cat.categories),
    dims=dict(theta=['subject']))
az.plot_trace(idata)
plt.tight_layout()

# +
fig, axes = plt.subplots(ncols=2, figsize=(8, 4))

ax = axes[0]
az.plot_posterior(idata, var_names='kappa',
                  point_estimate='mode', hdi_prob=.95, ax=ax)
ax.set_xlabel('$\\kappa$')

ax = axes[1]
az.plot_posterior(idata, var_names='omega',
                  point_estimate='mode', hdi_prob=.95, ref_val=0.5, ax=ax)
ax.set_title('Group Mode')
ax.set_xlabel('$\\omega$')

fig.tight_layout()

# +
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
theta_to_plot = ['S01', 'S14', 'S28']

for r, c in np.ndindex(axes.shape):
    ax = axes[r, c]
    r_theta = theta_to_plot[r]
    c_theta = theta_to_plot[c]

    if r == c:
        az.plot_posterior(
            idata, 'theta',
            coords=dict(subject=r_theta),
            hdi_prob=.95,
            point_estimate='mode',
            ref_val=.5,
            ax=ax)
        ax.set_title(f'theta[{r_theta}]')
    elif r < c:
        diff = (idata['posterior']['theta'].sel(subject=r_theta).values
                - idata['posterior']['theta'].sel(subject=c_theta).values)
        az.plot_posterior(
            diff, hdi_prob=.95, point_estimate='mode', ref_val=0, ax=ax)
        ax.set_title(f'theta[{r_theta}] - theta[{c_theta}]')
        ax.set_xlabel('Diff')
    else:
        y = idata['posterior']['theta'].sel(subject=r_theta).values.flatten()
        x = idata['posterior']['theta'].sel(subject=c_theta).values.flatten()

        sns.scatterplot(x=x, y=y, ax=ax)

        xx = np.linspace(*ax.get_xlim(), 1000)
        sns.lineplot(x=xx, y=xx, ax=ax, color='k', linestyle='dashed')

        ax.set_xlabel(f'theta[{c_theta}]')
        ax.set_ylabel(f'theta[{r_theta}]')

fig.tight_layout()
# -

# ## Extending the Hierarchy: Subjects within Categories
# ### Example: Baseball Batting Abilities by Position

baseball_df: pd.DataFrame = pd.read_csv(
    'datasets/BattingAverage.csv',
    dtype=dict(PriPos='category', Player='category'))
baseball_df.info()


# + tags=[]
def baseball_batting_model(y: jnp.ndarray, pos: jnp.ndarray, at_bats: jnp.ndarray, nb_pos: int):
    assert y.shape[0] == pos.shape[0] == at_bats.shape[0]
    nb_obs = y.shape[0]

    # All positions' overall ability.
    omega = numpyro.sample('omega', dist.Beta(1, 1))
    kappa_minus_two = numpyro.sample(
        '_kappa-2', dist_utils.gamma_dist_from_mean_std(1, 10))
    kappa = numpyro.deterministic('kappa', kappa_minus_two + 2)

    # Each position's ability.
    omega_pos = numpyro.sample(
        'omega_pos', dist_utils.beta_dist_from_omega_kappa(omega, kappa).expand([nb_pos]))
    kappa_pos_minus_two = numpyro.sample(
        '_kappa_pos-2', dist_utils.gamma_dist_from_mean_std(1, 10).expand([nb_pos]))
    kappa_pos = numpyro.deterministic('kappa_pos', kappa_pos_minus_two + 2)

    # Each player's ability.
    with numpyro.plate('obs', nb_obs) as idx:
        ability = numpyro.sample(
            'ability', dist_utils.beta_dist_from_omega_kappa(omega_pos[pos[idx]], kappa_pos[pos[idx]]))
        numpyro.sample('y', dist.Binomial(at_bats[idx], ability), obs=y[idx])


kernel = NUTS(baseball_batting_model)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(baseball_df['Hits'].values),
    pos=jnp.array(baseball_df['PriPos'].cat.codes.values),
    at_bats=jnp.array(baseball_df['AtBats'].values),
    nb_pos=baseball_df['PriPos'].cat.categories.size,
)
mcmc.print_summary()
# -

idata = az.from_numpyro(
    mcmc,
    coords=dict(
        position=baseball_df['PriPos'].cat.categories,
        player=baseball_df['Player'],
    ),
    dims=dict(
        omega_pos=['position'],
        kappa_pos=['position'],
        ability=['player'],
    ),
)
az.plot_trace(idata, var_names='~ability')
plt.tight_layout()


# +
def plot_position_ability(idata: az.InferenceData, positions: list[str]):
    nb_pos = len(positions)
    fig, axes = plt.subplots(nrows=nb_pos, ncols=nb_pos, figsize=(6, 6))

    omega = idata['posterior']['omega_pos']

    for r, c in np.ndindex(axes.shape):
        ax = axes[r, c]
        r_pos = positions[r]
        c_pos = positions[c]

        if r == c:
            az.plot_posterior(
                idata, var_names='omega_pos', coords=dict(position=r_pos),
                hdi_prob=.95, point_estimate='mode', ax=ax)
            ax.set_title(f'{r_pos}')
            ax.set_xlabel('$\\omega$')
        elif r < c:
            diff = (omega.sel(position=r_pos).values
                    - omega.sel(position=c_pos).values)
            az.plot_posterior(
                diff, hdi_prob=.95, point_estimate='mode', ref_val=0, ax=ax)
            ax.set_title(f'{r_pos} - {c_pos}')
            ax.set_xlabel('Diff. of $\\omega$')
        else:
            x = omega.sel(position=c_pos).values.flatten()
            y = omega.sel(position=r_pos).values.flatten()

            sns.scatterplot(x=x, y=y, ax=ax)
            ax.set_xlabel(f'{c_pos}')
            ax.set_ylabel(f'{r_pos}')

    fig.tight_layout()


plot_position_ability(idata, ['Pitcher', 'Catcher'])
plot_position_ability(idata, ['Catcher', '1st Base'])


# +
def plot_player_ability(idata: az.InferenceData, players: list[str]):
    nb_players = len(players)
    fig, axes = plt.subplots(
        nrows=nb_players, ncols=nb_players, figsize=(6, 6))

    ability = idata['posterior']['ability']

    for r, c in np.ndindex(axes.shape):
        ax = axes[r, c]
        r_player = players[r]
        c_player = players[c]

        if r == c:
            az.plot_posterior(
                idata, var_names='ability', coords=dict(player=r_player),
                hdi_prob=.95, point_estimate='mode', ax=ax)
            ax.set_title(f'{r_player}')
            ax.set_xlabel('$\\omega$')
        elif r < c:
            diff = (ability.sel(player=r_player).values
                    - ability.sel(player=c_player).values)
            az.plot_posterior(
                diff, hdi_prob=.95, point_estimate='mode', ref_val=0, ax=ax)
            ax.set_title(f'{r_player} - {c_player}')
            ax.set_xlabel('Diff. of $\\omega$')
        else:
            x = ability.sel(player=c_player).values.flatten()
            y = ability.sel(player=r_player).values.flatten()

            sns.scatterplot(x=x, y=y, ax=ax)
            ax.set_xlabel(f'{c_player}')
            ax.set_ylabel(f'{r_player}')

    fig.tight_layout()


plot_player_ability(idata, ['Kyle Blanks', 'Bruce Chen'])
plot_player_ability(idata, ['ShinSoo Choo', 'Ichiro Suzuki'])
# -

plot_player_ability(idata, ['Mike Leake', 'Wandy Rodriguez'])
plot_player_ability(idata, ['Andrew McCutchen', 'Brett Jackson'])

az.plot_posterior(idata, 'omega', hdi_prob=.95, point_estimate='mode')

# +
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))

for ax, position in zip(axes.flatten(), baseball_df['PriPos'].cat.categories):
    az.plot_posterior(
        idata,
        var_names='omega_pos',
        coords=dict(position=position),
        hdi_prob=.95,
        point_estimate='mode',
        ax=ax)
    ax.set_title(position)

fig.tight_layout()
