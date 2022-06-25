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
#     display_name: doing-bayes
#     language: python
#     name: doing-bayes
# ---

# %cd ../..
# %load_ext autoreload
# %autoreload 2

# +
import arviz as az
import numpyro.distributions as dist
import jax.numpy as jnp
import jax.random as random
from numpyro.infer.initialization import init_to_median
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
import numpyro_glm.ordinal as glm_ordinal

numpyro.set_host_device_count(4)


# -

# # Ordinal Model with Synthesis Data
# ## Synthesis Data

# +
def create_ranking_votes(mean: float, std: float, thresholds: 'list[float]', N: int) -> 'list[int]':
    points = np.random.normal(mean, std, N)
    thresholds = [-np.inf, ] + thresholds + [np.inf, ]
    votes = [np.sum((prev <= points) & (points <= cur))
             for cur, prev in zip(thresholds[1:], thresholds[:-1])]
    return votes


def plot_ranking_votes(votes: 'list[int]', labels: 'list[str]') -> None:
    fig, ax = plt.subplots()
    ax.bar(range(len(labels)), votes, width=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    fig.tight_layout()


LATENT_MEAN = 1.0
LATENT_STD = 1.0

N = 4970
ORDINAL_VALUES = ['0 star', '1 star',
                  '2 stars', '3 stars', '4 stars', '5 stars']
THRESHOLDS = [0.5, 1.5, 2.5, 3.5, 4.5]

votes = create_ranking_votes(LATENT_MEAN, LATENT_STD, THRESHOLDS, N)
plot_ranking_votes(votes, ORDINAL_VALUES)
# -

votes = pd.DataFrame(
    [['Prod', *votes]],
    columns=['Name', *ORDINAL_VALUES],
)
votes

# ## Model

data = votes.iloc[:, 1:].values
numpyro.render_model(
    glm_ordinal.one_group_1,
    model_args=(jnp.array(data), ORDINAL_VALUES),
    render_params=True,
)

mcmc_key = random.PRNGKey(1234)
kernel = NUTS(glm_ordinal.one_group_1, init_strategy=init_to_median)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(mcmc_key, jnp.array(data), ORDINAL_VALUES)
mcmc.print_summary()

idata = az.from_numpyro(mcmc)
az.plot_trace(idata)
plt.tight_layout()
