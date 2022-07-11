import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro_glm.utils.dist as dist_utils


def two_nominal_predictors(
        y: jnp.ndarray, x1: jnp.ndarray, x2: jnp.ndarray, K1: int, K2: int):
    """
    Count predicted model as described in Chapter 24, section 24.1.4
    """
    assert y.ndim == x1.ndim == x2.ndim == 1
    assert y.shape[0] == x1.shape[0] == x2.shape[0]

    nb_obs = y.shape[0]
    nb_cells = K1 * K2

    # Data statistics.
    ylog_mean = jnp.log(jnp.sum(y) / nb_cells)
    ylog_sd = jnp.log(jnp.std(jnp.r_[jnp.zeros(nb_cells), jnp.sum(y)]))

    # Prior for a0.
    a0 = numpyro.sample('_a0', dist.Normal(ylog_mean, 2 * ylog_sd))

    # Prior for a1.
    a1_sd = numpyro.sample(
        '_a1_sd', dist_utils.gammaDistFromModeStd(ylog_sd, ylog_sd * 2))
    a1 = numpyro.sample('_a1', dist.Normal(0, a1_sd).expand([K1]))

    # Prior for a2.
    a2_sd = numpyro.sample(
        '_a2_sd', dist_utils.gammaDistFromModeStd(ylog_sd, ylog_sd * 2))
    a2 = numpyro.sample('_a2', dist.Normal(0, a2_sd).expand([K2]))

    # Prior for interaction term.
    a1a2_sd = numpyro.sample(
        '_a1a2_sd', dist_utils.gammaDistFromModeStd(ylog_sd, ylog_sd * 2))
    a1a2 = numpyro.sample(
        '_a1a2', dist.Normal(0, a1a2_sd).expand([K1, K2]))

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        g1 = x1[idx]
        g2 = x2[idx]
        mean = jnp.exp(a0 + a1[g1] + a2[g2] + a1a2[g1, g2])
        numpyro.sample('y', dist.Poisson(mean), obs=y[idx])

    # Transform back to b to impose sum-to-zero constraint.
    m = a0 + a1[:, None] + a2[None, :] + a1a2
    b0 = numpyro.deterministic('b0', jnp.mean(m))
    b1 = numpyro.deterministic('b1', jnp.mean(m, axis=1) - b0)
    b2 = numpyro.deterministic('b2', jnp.mean(m, axis=0) - b0)
    numpyro.deterministic('b1b2', m - (b0 + b1[:, None] + b2[None, :]))

    # Compute predicted proportions.
    m_exp = jnp.exp(m)
    pp_x1x2 = numpyro.deterministic('P', m_exp / jnp.sum(m_exp))
    numpyro.deterministic('P_x1', jnp.sum(pp_x1x2, axis=1))
    numpyro.deterministic('P_x2', jnp.sum(pp_x1x2, axis=0))
