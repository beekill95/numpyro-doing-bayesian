import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def one_group(y: jnp.ndarray):
    n, = jnp.shape(y)
    data_mean = jnp.mean(y)
    data_std = jnp.std(y)

    # Specify prior mean and standard deviation.
    mean = numpyro.sample('mean', dist.Normal(data_mean, 100 * data_std))
    std = numpyro.sample('std', dist.Uniform(data_std / 1000, data_std * 1000))

    # Observations.
    with numpyro.plate('obs', n) as idx:
        numpyro.sample('y', dist.Normal(mean, std), obs=y[idx])


def one_group_robust(y: jnp.ndarray):
    n, = jnp.shape(y)
    data_mean = jnp.mean(y)
    data_std = jnp.std(y)

    # Specify prior mean, scale and normality parameter.
    mean = numpyro.sample('mean', dist.Normal(data_mean, 100 * data_std))
    sigma = numpyro.sample(
        'sigma', dist.Uniform(data_std / 1000, data_std * 1000))
    nu = numpyro.sample('nu', dist.Exponential(1. / 30))

    # Observations.
    with numpyro.plate('obs', n) as idx:
        numpyro.sample('y', dist.StudentT(nu, mean, sigma), obs=y[idx])


def multi_groups_robust(y: jnp.ndarray, group: jnp.ndarray, nb_groups: int):
    assert jnp.shape(y)[0] == jnp.shape(group)[0]

    n, = jnp.shape(y)
    data_mean = jnp.mean(y)
    data_std = jnp.std(y)

    # Specify priors for mean, std, and normality parameter.
    mean = numpyro.sample(
        'mean', dist.Normal(data_mean, 100 * data_std).expand((nb_groups, )))
    sigma = numpyro.sample(
        'sigma', dist.Uniform(data_std / 1000, data_std * 1000).expand((nb_groups, )))
    nu = numpyro.sample('nu', dist.Exponential(1. / 30))

    # Observations.
    with numpyro.plate('obs', n) as idx:
        numpyro.sample(
            'y', dist.StudentT(nu, mean[group[idx]], sigma[group[idx]]), obs=y[idx])


def one_metric_predictor_robust_no_standardization(y: jnp.ndarray, x: jnp.ndarray):
    assert jnp.shape(y)[0] == jnp.shape(x)[0]

    n, = jnp.shape(y)

    # Specify priors for b0, b1, sigma and normality parameter.
    b0 = numpyro.sample('b0', dist.Normal(0, 10))
    b1 = numpyro.sample('b1', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.Uniform(1e-3, 1e+3))
    nu = numpyro.sample('nu', dist.Exponential(1. / 30))
    mean = numpyro.deterministic('mean', b0 + x * b1)

    # Observations.
    with numpyro.plate('obs', n) as idx:
        numpyro.sample('y', dist.StudentT(nu, mean[idx], sigma), obs=y[idx])


def one_metric_predictor_robust(y: jnp.ndarray, x: jnp.ndarray):
    """The same one metric predictor model but with standardization."""
    assert jnp.shape(y)[0] == jnp.shape(x)[0]

    n, = jnp.shape(y)

    # Standardize the data.
    mean_x, mean_y = (jnp.mean(val) for val in [x, y])
    std_x, std_y = (jnp.std(val) for val in [x, y])
    xz = (x - mean_x) / std_x
    yz = (y - mean_y) / std_y

    # Specify priors zb0, zb1, zsigma and normality paramter.
    zb0 = numpyro.sample('zb0', dist.Normal(0, 10))
    zb1 = numpyro.sample('zb1', dist.Normal(0, 10))
    zsigma = numpyro.sample('zsigma', dist.Uniform(1e-3, 1e+3))
    nu = numpyro.sample('nu', dist.Exponential(1. / 30))
    zmean = numpyro.deterministic('zmean', zb0 + xz * zb1)

    # Observations.
    with numpyro.plate('zobs', n) as idx:
        numpyro.sample('yz', dist.StudentT(nu, zmean, zsigma), obs=yz[idx])

    # Transform back to the original scale.
    numpyro.deterministic('b1', zb1 * std_y / std_x)
    numpyro.deterministic(
        'b0', zb0 * std_y + mean_y - zb1 * mean_x * std_y / std_x)
    numpyro.deterministic('sigma', zsigma * std_y)
