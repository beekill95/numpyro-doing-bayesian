import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro_glm.utils import dist as dist_utils


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

    # Specify priors zb0, zb1, zsigma and normality parameter.
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


def multi_metric_predictors_robust(y: jnp.ndarray, x: jnp.ndarray):
    assert jnp.shape(y)[0] == jnp.shape(x)[0]
    n = jnp.shape(y)[0]
    n_preds = jnp.shape(x)[1]

    # Standardize and normalize the data.
    x_means = jnp.mean(x, axis=0)
    x_stds = jnp.std(x, axis=0)
    y_mean = jnp.mean(y)
    y_std = jnp.std(y)
    xz = (x - x_means) / x_stds
    yz = (y - y_mean) / y_std

    # Specify priors for coefficients, sigma and normality parameter.
    zb0 = numpyro.sample('zb0', dist.Normal(0, 10))
    zb_ = numpyro.sample('zb', dist.Normal(0, 10).expand((n_preds, )))
    zsigma = numpyro.sample('zsigma', dist.Uniform(1e-3, 1e3))
    nu = numpyro.sample('nu', dist.Exponential(1. / 30))

    # Observations.
    with numpyro.plate('zobs', n) as idx:
        zmean = zb0 + jnp.dot(xz[idx], zb_)
        numpyro.sample('yz', dist.StudentT(nu, zmean, zsigma), obs=yz)

    # Transform back to the original scale.
    numpyro.deterministic('b_', zb_ * y_std / x_stds)
    numpyro.deterministic(
        'b0', zb0 * y_std + y_mean - jnp.sum(zb_ * x_means / x_stds) * y_std)
    numpyro.deterministic('sigma', zsigma * y_std)


def multi_metric_predictors_robust_with_shrinkage(y: jnp.ndarray, x: jnp.ndarray):
    assert jnp.shape(y)[0] == jnp.shape(x)[0]
    n = jnp.shape(y)[0]
    n_preds = jnp.shape(x)[1]

    # Standardize and normalize the data.
    x_means = jnp.mean(x, axis=0)
    x_stds = jnp.std(x, axis=0)
    y_mean = jnp.mean(y)
    y_std = jnp.std(y)
    xz = (x - x_means) / x_stds
    yz = (y - y_mean) / y_std

    # Specify priors for shrinkage of coefficients.
    sigma_b_ = numpyro.sample('sigma_b_', dist.Gamma(1.0, 1.0))

    # Specify priors for coefficients, sigma and normality parameter.
    zb0 = numpyro.sample('zb0', dist.Normal(0, 10))
    zb_ = numpyro.sample(
        'zb_', dist.StudentT(1, 0, sigma_b_).expand((n_preds, )))
    zsigma = numpyro.sample('zsigma', dist.Uniform(1e-3, 1e3))
    nu = numpyro.sample('nu', dist.Exponential(1. / 30))

    # Observations.
    with numpyro.plate('zobs', n) as idx:
        zmean = zb0 + jnp.dot(xz[idx], zb_)
        numpyro.sample('yz', dist.StudentT(nu, zmean, zsigma), obs=yz)

    # Transform back to the original scale.
    numpyro.deterministic('b_', zb_ * y_std / x_stds)
    numpyro.deterministic(
        'b0', zb0 * y_std + y_mean - jnp.sum(zb_ * x_means / x_stds) * y_std)
    numpyro.deterministic('sigma', zsigma * y_std)


def multi_metric_predictors_robust_with_selection(y: jnp.ndarray, x: jnp.ndarray):
    assert jnp.shape(y)[0] == jnp.shape(x)[0]
    n = jnp.shape(y)[0]
    n_preds = jnp.shape(x)[1]

    # Standardize and normalize the data.
    x_means = jnp.mean(x, axis=0)
    x_stds = jnp.std(x, axis=0)
    y_mean = jnp.mean(y)
    y_std = jnp.std(y)
    xz = (x - x_means) / x_stds
    yz = (y - y_mean) / y_std

    # Specify inclusion parameters.
    included = numpyro.sample('delta', dist.Bernoulli(0.5).expand((n_preds, )))

    sigma_b_ = numpyro.sample('sigmab', dist.Gamma(1.1051, 0.1051))

    # Specify priors for coefficients, sigma and normality parameter.
    zb0 = numpyro.sample('zb0', dist.Normal(0, 2))
    # zb_ = numpyro.sample('zb', dist.Normal(0, 2).expand((n_preds, )))
    zb_ = numpyro.sample('zb_', dist.StudentT(
        1, 0, sigma_b_).expand((n_preds, )))
    zsigma = numpyro.sample('zsigma', dist.Uniform(1e-5, 10))
    nu = numpyro.sample('nu', dist.Exponential(1. / 30))

    # Observations.
    with numpyro.plate('zobs', n) as idx:
        zmean = zb0 + jnp.dot(xz[idx], zb_ * included)
        numpyro.sample('yz', dist.StudentT(nu, zmean, zsigma), obs=yz)

    # Transform back to the original scale.
    numpyro.deterministic('b_', included * zb_ * y_std / x_stds)
    numpyro.deterministic(
        'b0', zb0 * y_std + y_mean - jnp.sum(included * zb_ * x_means / x_stds) * y_std)
    numpyro.deterministic('sigma', zsigma * y_std)


def hierarchical_one_metric_predictor_multi_groups_robust(
        y: jnp.ndarray, x: jnp.ndarray, group: jnp.ndarray, nb_groups: int):
    assert jnp.shape(y)[0] == jnp.shape(x)[0] == jnp.shape(group)[0]

    n = jnp.shape(y)[0]
    x_mean, y_mean = (jnp.mean(v) for v in [x, y])
    x_std, y_std = (jnp.std(v) for v in [x, y])

    # Data standardization.
    xz = (x - x_mean) / x_std
    yz = (y - y_mean) / y_std

    # Specify mean and sigma priors of zb0 and zb1.
    zb0_mean = numpyro.sample('zb0_mean', dist.Normal(0, 10))
    zb1_mean = numpyro.sample('zb1_mean', dist.Normal(0, 10))
    zb0_std = numpyro.sample('zb0_std', dist.Uniform(1e-3, 1e3))
    zb1_std = numpyro.sample('zb1_std', dist.Uniform(1e-3, 1e3))

    # Specify the distribution of zb0 and zb1.
    # These priors are sampled in Normal(0, 1) space,
    # and then multiply with mean and std to get Normal(mean, std)
    # to prevent divergences in NUTS sampling.
    zb0_ = numpyro.sample('zb0_', dist.Normal(0, 1).expand((nb_groups, )))
    zb0 = numpyro.deterministic('zb0', zb0_ * zb0_std + zb0_mean)
    zb1_ = numpyro.sample('zb1_', dist.Normal(0, 1).expand((nb_groups, )))
    zb1 = numpyro.deterministic('zb1', zb1_ * zb1_std + zb1_mean)

    # Specify zsigma and normality priors.
    zsigma = numpyro.sample('zsigma', dist.Uniform(1e-3, 1e3))
    nu = numpyro.sample('nu', dist.Exponential(1. / 30))

    # Observations.
    with numpyro.plate('zobs', n) as idx:
        zmean = zb0[group[idx]] + xz[idx] * zb1[group[idx]]
        numpyro.sample('yz', dist.StudentT(nu, zmean, zsigma), obs=yz[idx])

    # Transform back to original scale.
    numpyro.deterministic('b1', zb1 * y_std / x_std)
    numpyro.deterministic(
        'b0', zb0 * y_std + y_mean - zb1 * x_mean * y_std / x_std)
    numpyro.deterministic('b1_mean', zb1_mean * y_std / x_std)
    numpyro.deterministic(
        'b0_mean', zb0_mean * y_std + y_mean - zb1_mean * x_mean * y_std / x_std)
    numpyro.deterministic('sigma', zsigma * y_std)


def hierarchical_quadtrend_one_metric_predictor_multi_groups_robust(
        y: jnp.ndarray, x: jnp.ndarray, group: jnp.ndarray, nb_groups: int, y_noise: jnp.ndarray = None):
    assert (jnp.shape(y)[0]
            == jnp.shape(x)[0]
            == jnp.shape(group)[0]
            == jnp.shape(y_noise)[0])

    n = jnp.shape(y)[0]
    x_mean, y_mean = (jnp.mean(v) for v in [x, y])
    x_std, y_std = (jnp.std(v) for v in [x, y])

    if y_noise is not None:
        y_noise_z = y_noise / jnp.mean(y_noise)
    else:
        y_noise_z = jnp.ones(n)

    # Standardize data.
    xz = (x - x_mean) / x_std
    yz = (y - y_mean) / y_std

    # Specify mean and sigma priors of b0_z, b1_z, and b2_z.
    b0_z_mean = numpyro.sample('b0_z_mean', dist.Normal(0, 10))
    b1_z_mean = numpyro.sample('b1_z_mean', dist.Normal(0, 10))
    b2_z_mean = numpyro.sample('b2_z_mean', dist.Normal(0, 10))
    b0_z_std = numpyro.sample('b0_z_std', dist.Uniform(1e-3, 1e3))
    b1_z_std = numpyro.sample('b1_z_std', dist.Uniform(1e-3, 1e3))
    b2_z_std = numpyro.sample('b2_z_std', dist.Uniform(1e-3, 1e3))

    # Specify the distribution of b0_z, b1_z, and b2_z.
    b0_z_ = numpyro.sample('b0_z_', dist.Normal(0, 1).expand((nb_groups, )))
    b1_z_ = numpyro.sample('b1_z_', dist.Normal(0, 1).expand((nb_groups, )))
    b2_z_ = numpyro.sample('b2_z_', dist.Normal(0, 1).expand((nb_groups, )))
    b0_z = numpyro.deterministic('b0_z', b0_z_ * b0_z_std + b0_z_mean)
    b1_z = numpyro.deterministic('b1_z', b1_z_ * b1_z_std + b1_z_mean)
    b2_z = numpyro.deterministic('b2_z', b2_z_ * b2_z_std + b2_z_mean)

    # Specify the priors of sigma and normality parameter.
    sigma_z = numpyro.sample('sigma_z', dist.Uniform(1e-3, 1e3))
    nu = numpyro.sample('nu', dist.Exponential(1. / 30))

    # Observations.
    with numpyro.plate('obs_z', n) as idx:
        mean_z = (b0_z[group[idx]]
                  + xz[idx] * b1_z[group[idx]]
                  + xz[idx]**2 * b2_z[group[idx]])
        numpyro.sample(
            'yz_obs', dist.StudentT(nu, mean_z, y_noise_z[idx] * sigma_z), obs=yz[idx])

    # Transform back to the original scale.
    numpyro.deterministic('b2', b2_z * y_std / (x_std**2))
    numpyro.deterministic(
        'b1', b1_z * y_std / x_std - 2 * b2_z * x_mean * y_std / (x_std**2))
    numpyro.deterministic(
        'b0', (b0_z * y_std
               + y_mean
               - b1_z * x_mean * y_std / x_std
               + b2_z * x_mean**2 * y_std / (x_std**2)))
    numpyro.deterministic('b2_mean', b2_z_mean * y_std / (x_std**2))
    numpyro.deterministic(
        'b1_mean', b1_z_mean * y_std / x_std - 2 * b2_z_mean * x_mean * y_std / (x_std**2))
    numpyro.deterministic(
        'b0_mean', (b0_z_mean * y_std
                    + y_mean
                    - b1_z_mean * x_mean * y_std / x_std
                    + b2_z_mean * x_mean**2 * y_std / (x_std**2)))
    numpyro.deterministic('sigma', sigma_z * y_std)


def one_nominal_predictor(y: jnp.ndarray, x: jnp.ndarray, nb_groups: int):
    """
    Bayesian model as explained in Chapter 19, Section 19.3, Figure 19.2

    Parameters
    ----------
    y: jnp.ndarray
        Metric predicted.
    x: jnp.ndarray
        Nominal predictor, integer values show which group does the data belong to.
    nb_groups: int
        Number of different groups existed in `x`.
    """
    assert y.shape[0] == x.shape[0]
    assert x.ndim == 1

    nb_obs = y.shape[0]

    # Calculate data statistics.
    y_mean = jnp.mean(y)
    y_std = jnp.std(y)

    # Specify priors.
    a0 = numpyro.sample('a0', dist.Normal(y_mean, y_std * 5))

    a_sigma = numpyro.sample(
        'a_sigma', dist_utils.gammaDistFromModeStd(y_std / 2, 2 * y_std))
    a_ = numpyro.sample('a_', dist.Normal(0, a_sigma).expand((nb_groups, )))

    ySigma = numpyro.sample('ySigma', dist.Uniform(y_std / 100, y_std * 10))

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        mean = a0 + a_[x[idx]]
        numpyro.sample('y', dist.Normal(mean, ySigma), obs=y[idx])

    # Transform to the actual intercept and coefficients
    # by imposing sum-to-zero constraints on `a_`.
    m = a0 + a_
    b0 = numpyro.deterministic('b0', jnp.mean(m))
    numpyro.deterministic('b_', m - b0)
