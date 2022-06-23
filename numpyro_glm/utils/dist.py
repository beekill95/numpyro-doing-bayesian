import jax.numpy as jnp
import numpyro.distributions as dist


def gammaDistFromModeStd(mode, std):
    # assert mode > 0, 'Mode must be positive'
    # assert std > 0, 'Standard must be positive'

    std_squared = std**2
    rate = (mode + jnp.sqrt(mode**2 + 4 * std_squared)) / (2 * std_squared)
    shape = 1 + mode * rate

    return dist.Gamma(shape, rate)


def beta_dist_from_omega_kappa(omega, kappa):
    return dist.Beta(omega * (kappa - 2) + 1, (1 - omega) * (kappa - 2) + 1)
