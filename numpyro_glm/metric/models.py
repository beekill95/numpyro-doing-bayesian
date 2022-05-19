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


def multi_groups_robust(y: jnp.ndarray, x: jnp.ndarray):
    assert jnp.allclose(jnp.shape(y), jnp.shape(x))

    n, = jnp.shape(y)
    nb_groups = len(jnp.unique(x))
    data_mean = jnp.mean(y)
    data_std = jnp.std(y)

    # Specify priors for mean, std, and normality parameter.
    mean = numpyro.sample(
        'mean', dist.Normal(data_mean, 100 * data_std).expand(nb_groups))
    sigma = numpyro.sample(
        'sigma', dist.Uniform(data_std / 1000, data_std * 1000).expand(nb_groups))
    nu = numpyro.sample('nu', dist.Exponential(1. / 30))

    # Observations.
    with numpyro.plate('obs', n) as idx:
        numpyro.sample(
            'y', dist.StudentT(nu, mean[x[idx]], sigma[x[idx]]), obs=y[idx])
