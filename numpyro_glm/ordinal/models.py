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
    cdf = jnp.r_[0., cdf, 1.]

    # From the thresholds, calculate the probability for each category.
    probs = jnp.diff(cdf)
    probs = jnp.maximum(probs, 0.)
    probs = probs / jnp.sum(probs)

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        numpyro.sample('y', dist.Categorical(probs), obs=y[idx])


def yord_two_groups(y: jnp.ndarray, grp: jnp.ndarray, K: int, nb_groups: int):
    """
    Two group ordinal model as described in Chapter 23, section 23.3.

    Parameters
    ----------
    y: jnp.ndarray
        Ordinal predicted variable.
    grp: jnp.ndarray
        Tell which group a data point belongs to.
    K: int
        Number of different ordinal outcomes.
    nb_groups: int
        Number of different groups in `grp`.
    """
    assert y.shape[0] == grp.shape[0]
    assert nb_groups == 2  # although, this could be removed!

    nb_obs = y.shape[0]

    # Specify mean and scale of latent score.
    mu = numpyro.sample(
        'mu', dist.Normal((K + 1.) / 2, K).expand([nb_groups]))
    sigma = numpyro.sample(
        'sigma', dist.Uniform(K / 1000., K * 10.).expand([nb_groups]))
    score = dist.Normal(mu[:, None], sigma[:, None])

    # Specify the thresholds.
    thres = jnp.array([
        numpyro.deterministic('thres_1', jnp.array(1.5)),
        *[numpyro.sample(f'thres_{i + 1}', dist.Normal(i + 1.5, 2))
          for i in range(1, K - 2)],
        numpyro.deterministic(f'thres_{K - 1}', jnp.array(K - 0.5)),
    ])
    cdf = score.cdf(thres[None, :])
    cdf = jnp.c_[
        jnp.repeat(jnp.array([0.]), nb_groups),
        cdf,
        jnp.repeat(jnp.array([1.]), nb_groups)]

    # Calculate probability for each group and each outcome.
    probs = jnp.diff(cdf, axis=-1)
    probs = jnp.maximum(probs, 0.)
    probs = probs / jnp.sum(probs, axis=1, keepdims=True)

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        numpyro.sample('y', dist.Categorical(probs[grp[idx]]), obs=y[idx])


def yord_metric_predictors(y: jnp.ndarray, x: jnp.ndarray, K: int):
    """
    Probit regression model as described in chapter 23, section 23.4.
    """
    assert y.shape[0] == x.shape[0]
    assert x.ndim == 2

    nb_obs = y.shape[0]
    nb_preds = x.shape[1]

    # Data statistics.
    x_means = jnp.mean(x, axis=0)
    x_sds = jnp.std(x, axis=0)
    xz = (x - x_means) / x_sds

    # Prior for intercept and coefficients.
    a0 = numpyro.sample('_a0', dist.Normal((K + 1.) / 2, K))
    a = numpyro.sample('_a', dist.Normal(0, K).expand([nb_preds]))

    # Latent mean and sigma.
    mu = numpyro.deterministic('mu', a0 + jnp.dot(xz, a))
    sigma = numpyro.sample('sigma', dist.Uniform(K / 1000, K * 10))
    score = dist.Normal(mu[:, None], sigma)

    # Specify the thresholds.
    thres = jnp.array([
        numpyro.deterministic('thres_1', jnp.array(1.5)),
        *[numpyro.sample(f'thres_{i + 1}', dist.Normal(i + 1.5, 2))
          for i in range(1, K - 2)],
        numpyro.deterministic(f'thres_{K - 1}', jnp.array(K - 0.5)),
    ])
    cdf = score.cdf(thres[None, :])
    cdf = jnp.c_[
        jnp.repeat(jnp.array([0.]), nb_obs),
        cdf,
        jnp.repeat(jnp.array([1.]), nb_obs),
    ]

    # Probability.
    probs = jnp.diff(cdf, axis=-1)
    probs = jnp.maximum(probs, 0.)
    probs = probs / jnp.sum(probs, axis=1, keepdims=True)

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        numpyro.sample('y', dist.Categorical(probs[idx]), obs=y[idx])

    # Transform back to b.
    numpyro.deterministic('b', a / x_sds)
    numpyro.deterministic('b0', a0 - jnp.sum(a * x_means / x_sds))


def yord_metric_predictors_robust_guessing(y: jnp.ndarray, x: jnp.ndarray, K: int):
    """
    Probit regression model as described in chapter 23, section 23.4.
    In addition, this model includes a random guessing mixture to account for outliers.
    Model description can be found in exercise 23.2.
    """
    assert y.shape[0] == x.shape[0]
    assert x.ndim == 2

    nb_obs = y.shape[0]
    nb_preds = x.shape[1]

    # Data statistics.
    x_means = jnp.mean(x, axis=0)
    x_sds = jnp.std(x, axis=0)
    xz = (x - x_means) / x_sds

    # Prior for intercept and coefficients.
    a0 = numpyro.sample('_a0', dist.Normal((K + 1.) / 2, K))
    a = numpyro.sample('_a', dist.Normal(0, K).expand([nb_preds]))

    # Latent mean and sigma.
    mu = numpyro.deterministic('mu', a0 + jnp.dot(xz, a))
    sigma = numpyro.sample('sigma', dist.Uniform(K / 1000, K * 10))
    score = dist.Normal(mu[:, None], sigma)

    # Specify the thresholds.
    thres = jnp.array([
        numpyro.deterministic('thres_1', jnp.array(1.5)),
        *[numpyro.sample(f'thres_{i + 1}', dist.Normal(i + 1.5, 2))
          for i in range(1, K - 2)],
        numpyro.deterministic(f'thres_{K - 1}', jnp.array(K - 0.5)),
    ])
    cdf = score.cdf(thres[None, :])
    cdf = jnp.c_[
        jnp.repeat(jnp.array([0.]), nb_obs),
        cdf,
        jnp.repeat(jnp.array([1.]), nb_obs),
    ]

    # Probability.
    probs = jnp.diff(cdf, axis=-1)
    probs = jnp.maximum(probs, 0.)
    probs = probs / jnp.sum(probs, axis=1, keepdims=True)

    # Guessing parameter.
    alpha = numpyro.sample('alpha', dist.Beta(1, 9))
    guess = jnp.ones(K) / K

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        pr = (1 - alpha) * probs[idx] + alpha * guess
        numpyro.sample('y', dist.Categorical(pr), obs=y[idx])

    # Transform back to b.
    numpyro.deterministic('b', a / x_sds)
    numpyro.deterministic('b0', a0 - jnp.sum(a * x_means / x_sds))


def yord_metric_predictors_robust_t_dist(y: jnp.ndarray, x: jnp.ndarray, K: int):
    """
    Probit regression model as described in chapter 23, section 23.4.
    """
    assert y.shape[0] == x.shape[0]
    assert x.ndim == 2

    nb_obs = y.shape[0]
    nb_preds = x.shape[1]

    # Data statistics.
    x_means = jnp.mean(x, axis=0)
    x_sds = jnp.std(x, axis=0)
    xz = (x - x_means) / x_sds

    # Prior for intercept and coefficients.
    a0 = numpyro.sample('_a0', dist.Normal((K + 1.) / 2, K))
    a = numpyro.sample('_a', dist.Normal(0, K).expand([nb_preds]))

    # Latent mean, sigma and normality parameter.
    mu = numpyro.deterministic('mu', a0 + jnp.dot(xz, a))
    sigma = numpyro.sample('sigma', dist.Uniform(K / 1000, K * 10))
    nu = numpyro.sample('nu', dist.Exponential(1. / 30))
    score = dist.StudentT(nu, mu[:, None], sigma)

    # Specify the thresholds.
    thres = jnp.array([
        numpyro.deterministic('thres_1', jnp.array(1.5)),
        *[numpyro.sample(f'thres_{i + 1}', dist.Normal(i + 1.5, 2))
          for i in range(1, K - 2)],
        numpyro.deterministic(f'thres_{K - 1}', jnp.array(K - 0.5)),
    ])
    cdf = score.cdf(thres[None, :])
    cdf = jnp.c_[
        jnp.repeat(jnp.array([0.]), nb_obs),
        cdf,
        jnp.repeat(jnp.array([1.]), nb_obs),
    ]

    # Probability.
    probs = jnp.diff(cdf, axis=-1)
    probs = jnp.maximum(probs, 0.)
    probs = probs / jnp.sum(probs, axis=1, keepdims=True)

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        numpyro.sample('y', dist.Categorical(probs[idx]), obs=y[idx])

    # Transform back to b.
    numpyro.deterministic('b', a / x_sds)
    numpyro.deterministic('b0', a0 - jnp.sum(a * x_means / x_sds))
