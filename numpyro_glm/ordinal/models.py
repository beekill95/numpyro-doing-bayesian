import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def yord_single_group(y: jnp.ndarray, K: int):
    """
    Single group model as described in Chapter 23, Section 23.2.

    Parameters
    ----------
    y: jnp.ndarray
        Ordinal output.
    K: int
        Number of different ordinal categories in `y`.
    """
    assert y.ndim == 1

    nb_obs = y.shape[0]

    # Specify mean and scale of latent score.
    mu = numpyro.sample('mu', dist.Normal((K + 1.) / 2, K))
    sigma = numpyro.sample('sigma', dist.Uniform(K / 1000., K * 10.))
    score = dist.Normal(mu, sigma)

    # Specify the thresholds.
    thres = jnp.array([
        numpyro.deterministic('thres_1', jnp.array(1.5)),
        *[numpyro.sample(f'thres_{i + 1}', dist.Normal(i + 1.5, 2))
          for i in range(1, K - 2)],
        numpyro.deterministic(f'thres_{K - 1}', jnp.array(K - 0.5)),
    ])
    cdf = score.cdf(thres)

    # From the thresholds, calculate the probability for each category.
    probs = jnp.zeros(K, dtype=jnp.float32)
    probs = probs.at[0].set(cdf[0])
    probs = probs.at[jnp.arange(1, K - 1)].set(cdf[1:] - cdf[:-1])
    probs = probs.at[-1].set(1. - cdf[-1])
    probs = jnp.maximum(probs, 0.)
    probs = probs / jnp.sum(probs)

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        numpyro.sample('y', dist.Categorical(probs), obs=y[idx])
