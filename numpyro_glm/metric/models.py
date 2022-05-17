import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def one_group(y: jnp.array):
    n, = jnp.shape(y)
    data_mean = jnp.mean(y)
    data_std = jnp.std(y)
    
    # Specify prior mean and standard deviation.
    mean = numpyro.sample('mean', dist.Normal(data_mean, 100 * data_std))
    std = numpyro.sample('std', dist.Uniform(data_std / 1000, data_std * 1000))

    # Observations.
    with numpyro.plate('obs', n) as idx:
        numpyro.sample('y', dist.Normal(mean, std), obs=y[idx])