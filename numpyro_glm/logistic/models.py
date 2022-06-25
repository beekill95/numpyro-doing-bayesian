import jax.numpy as jnp
from jax.scipy.special import expit
import numpyro
import numpyro.distributions as dist
import numpyro_glm.utils.dist as dist_utils


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


def binom_one_nominal_predictor(y: jnp.ndarray, grp: jnp.ndarray, N: jnp.ndarray, nb_groups: int):
    """
    Binomial predicted variable with one nominal predictor model
    as described in chapter 21, section 21.4.2, figure 21.12.
    """
    assert y.shape[0] == grp.shape[0] == N.shape[0]

    nb_obs = y.shape[0]

    # Prior for intercept.
    a0 = numpyro.sample('_a0', dist.Normal(0, 2))

    # Prior for coefficients.
    a_sigma = numpyro.sample('a_sigma', dist_utils.gammaDistFromModeStd(2, 4))
    a = numpyro.sample('_a', dist.Normal(0, a_sigma).expand((nb_groups, )))

    # Prior for kappa of Beta distribution.
    kappa_minus_2 = numpyro.sample(
        '_kappa_minus_2', dist_utils.gammaDistFromModeStd(1, 10))
    kappa = numpyro.deterministic('kappa', kappa_minus_2 + 2)

    # Specify mu distribution.
    omega = numpyro.deterministic('omega', expit(a0 + a))

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        mu = numpyro.sample(
            'mu', dist_utils.beta_dist_from_omega_kappa(omega[grp[idx]], kappa))
        numpyro.sample('y', dist.Binomial(N[idx], mu), obs=y[idx])

    # Convert from a to b to impose sum-to-zero constraint.
    m = a0 + a
    b0 = numpyro.deterministic('b0', jnp.mean(m))
    numpyro.deterministic('b', m - b0)


def binom_one_nominal_predictor_het(y: jnp.ndarray, grp: jnp.ndarray, N: jnp.ndarray, nb_groups: int):
    """
    Binomial predicted variable with one nominal predictor model
    as described in chapter 21, section 21.4.2, figure 21.12.
    """
    assert y.shape[0] == grp.shape[0] == N.shape[0]

    nb_obs = y.shape[0]

    # Prior for intercept.
    a0 = numpyro.sample('_a0', dist.Normal(0, 2))

    # Prior for coefficients.
    a_sigma = numpyro.sample('a_sigma', dist_utils.gammaDistFromModeStd(2, 4))
    a = numpyro.sample('_a', dist.Normal(0, a_sigma).expand((nb_groups, )))

    # Prior for kappa of Beta distribution.
    kappa_minus_2 = numpyro.sample(
        '_kappa_minus_2', dist_utils.gammaDistFromModeStd(1, 10).expand((nb_groups, )))
    kappa = numpyro.deterministic('kappa', kappa_minus_2 + 2)

    # Specify mu distribution.
    omega = numpyro.deterministic('omega', expit(a0 + a))

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        mu = numpyro.sample(
            'mu', dist_utils.beta_dist_from_omega_kappa(omega[grp[idx]], kappa[grp[idx]]))
        numpyro.sample('y', dist.Binomial(N[idx], mu), obs=y[idx])

    # Convert from a to b to impose sum-to-zero constraint.
    m = a0 + a
    b0 = numpyro.deterministic('b0', jnp.mean(m))
    numpyro.deterministic('b', m - b0)


def softmax_multi_metric_predictors(y: jnp.ndarray, x: jnp.ndarray):
    """
    Parameters
    ----------
    """
    assert y.shape[0] == x.shape[0]
    assert x.ndim == 2

    nb_obs = y.shape[0]
    nb_preds = x.shape[1]

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
