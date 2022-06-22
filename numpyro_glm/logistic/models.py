import jax.numpy as jnp
from jax.scipy.special import expit
import numpyro
import numpyro.distributions as dist


def dich_multi_metric_predictors(y: jnp.ndarray, x: jnp.ndarray):
    """
    Dichotomous predicted model with multiple metric predictors model
    as described in chapter 21, section 21.1, figure 21.2.

    Parameters
    ----------
    y: jnp.ndarray
        a dichotomous predicted variable.
    x: jnp.ndarray
        metric predictors.
    """
    assert y.shape[0] == x.shape[0]
    assert x.ndim == 2

    nb_obs = y.shape[0]
    nb_pred = x.shape[1]

    # Metric predictors statistics.
    x_mean = jnp.mean(x, axis=0)
    x_sd = jnp.std(x, axis=0)

    # Normalize x.
    x_z = (x - x_mean) / x_sd

    # Specify priors for intercept term.
    _a0 = numpyro.sample('_a0', dist.Normal(0, 2))

    # Specify priors for coefficient terms.
    _a = numpyro.sample('_a', dist.Normal(0, 2).expand((nb_pred, )))

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        logit = _a0 + jnp.dot(x_z[idx], _a)
        numpyro.sample('y', dist.BernoulliLogits(logit), obs=y[idx])

    # Transform back to beta.
    numpyro.deterministic('b0', _a0 - jnp.dot(_a, x_mean / x_sd))
    numpyro.deterministic('b', _a / x_sd)


def dich_multi_metric_predictors_robust(y: jnp.ndarray, x: jnp.ndarray):
    """
    Robust dichotomous predicted model with multiple metric predictors model
    as described in chapter 21, section 21.3.

    Parameters
    ----------
    y: jnp.ndarray
        a dichotomous predicted variable.
    x: jnp.ndarray
        metric predictors.
    """
    assert y.shape[0] == x.shape[0]
    assert x.ndim == 2

    nb_obs = y.shape[0]
    nb_pred = x.shape[1]

    # Metric predictors statistics.
    x_mean = jnp.mean(x, axis=0)
    x_sd = jnp.std(x, axis=0)

    # Normalize x.
    x_z = (x - x_mean) / x_sd

    # Specify priors for intercept term.
    _a0 = numpyro.sample('_a0', dist.Normal(0, 2))

    # Specify priors for coefficient terms.
    _a = numpyro.sample('_a', dist.Normal(0, 2).expand((nb_pred, )))

    # Prior for the guess term.
    guess = numpyro.sample('guess', dist.Beta(1, 9))

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        lin_core = _a0 + jnp.dot(x_z[idx], _a)
        prob = guess * 0.5 + (1 - guess) * expit(lin_core)
        numpyro.sample('y', dist.Bernoulli(prob), obs=y[idx])

    # Transform back to beta.
    numpyro.deterministic('b0', _a0 - jnp.dot(_a, x_mean / x_sd))
    numpyro.deterministic('b', _a / x_sd)
