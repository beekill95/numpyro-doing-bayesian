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


def one_nominal_predictor(y: jnp.ndarray, grp: jnp.ndarray, nb_groups: int):
    """
    Bayesian model as explained in Chapter 19, Section 19.3, Figure 19.2

    Parameters
    ----------
    y: jnp.ndarray
        Metric predicted.
    grp: jnp.ndarray
        Nominal predictor, integer values show which group does the data belong to.
    nb_groups: int
        Number of unique groups in `grp`.
    """
    assert y.shape[0] == grp.shape[0]
    assert grp.ndim == 1

    nb_obs = y.shape[0]

    # Calculate data statistics.
    y_mean = jnp.mean(y)
    y_std = jnp.std(y)

    # Specify priors.
    a0 = numpyro.sample('a0', dist.Normal(y_mean, y_std * 5))

    a_sigma = numpyro.sample(
        'a_sigma', dist_utils.gammaDistFromModeStd(y_std / 2, 2 * y_std))
    a_ = numpyro.sample('a_grp', dist.Normal(0, a_sigma).expand((nb_groups, )))

    ySigma = numpyro.sample('y_sigma', dist.Uniform(y_std / 100, y_std * 10))

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        mean = a0 + a_[grp[idx]]
        numpyro.sample('y', dist.Normal(mean, ySigma), obs=y[idx])

    # Transform to the actual intercept and coefficients
    # by imposing sum-to-zero constraints on `a_`.
    m = a0 + a_
    b0 = numpyro.deterministic('b0', jnp.mean(m))
    numpyro.deterministic('b_grp', m - b0)


def one_nominal_one_metric(y: jnp.ndarray, grp: jnp.ndarray, cov: jnp.ndarray, nb_groups: int):
    """
    Bayesian model as described in Section 19.4, figure 19.4

    Parameters
    ----------
    y: jnp.ndarray
        Metric Predicted.
    grp: jnp.ndarray
        Nominal predictor, indicates which group does the data point belongs to.
    cov: jnp.ndarray
        Metric predictor.
    nb_groups: int
        Number of unique groups in `grp`
    """
    assert y.shape[0] == grp.shape[0] == cov.shape[0]
    assert y.ndim == grp.ndim == cov.ndim == 1

    nb_obs = y.shape[0]

    # Statistics of predicted.
    y_mean = jnp.mean(y)
    y_sd = jnp.std(y)

    # Statistics of metric predictor.
    cov_mean = jnp.mean(cov)
    cov_sd = jnp.std(cov)

    # Specify priors of `a` intercept and coefficients
    # (no imposing sum-to-zero constraint).
    a0 = numpyro.sample('a0', dist.Normal(y_mean, y_sd * 5))

    a_grp_sigma = numpyro.sample(
        'a_grp_sigma', dist_utils.gammaDistFromModeStd(y_sd / 2, y_sd * 2))
    a_grp = numpyro.sample(
        'a_grp', dist.Normal(0, a_grp_sigma).expand((nb_groups, )))

    a_cov = numpyro.sample('a_cov', dist.Normal(0, 2 * y_sd / cov_sd))

    y_sigma = numpyro.sample('y_sigma', dist.Uniform(y_sd / 100, y_sd * 100))

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        mean = a0 + a_grp[grp[idx]] + a_cov * (cov[idx] - cov_mean)
        numpyro.sample('y', dist.Normal(mean, y_sigma), obs=y[idx])

    # Convert from `a` to `b` to impose sum-to-zero constraint.
    a_grp_mean = jnp.mean(a_grp)
    numpyro.deterministic('b0', a0 + a_grp_mean - a_cov * cov_mean)
    numpyro.deterministic('b_grp', a_grp - a_grp_mean)
    numpyro.deterministic('b_cov', a_cov)


def one_nominal_predictor_het_var_robust(y: jnp.ndarray, grp: jnp.ndarray, nb_groups: int):
    """
    Bayesian model described in Section 19.5, figure 19.6

    Parameters
    ----------
    y: jnp.ndarray
        Metric predicted.
    grp: jnp.ndarray
        Nominal predictor: the group that the data point belongs to.
    nb_groups: int
        Number of unique groups in `grp`.
    """
    assert y.shape[0] == grp.shape[0]
    assert y.ndim == grp.ndim == 1

    nb_obs = y.shape[0]

    # Predicted's statistics.
    y_mean = jnp.mean(y)
    y_sd = jnp.std(y)

    # Priors of the linear coefficients.
    a_grp_sigma = numpyro.sample(
        'a_grp_sigma', dist_utils.gammaDistFromModeStd(y_sd / 2, y_sd * 2))

    a0 = numpyro.sample('a0', dist.Normal(y_mean, y_sd * 5))
    a_grp = numpyro.sample(
        'a_grp', dist.Normal(0, a_grp_sigma).expand((nb_groups, )))

    # Normality parameter.
    nu = numpyro.sample('nu', dist.Exponential(1. / 30))

    # Prior for y_sigma.
    y_sigma_mode = numpyro.sample(
        'y_sigma_mode', dist_utils.gammaDistFromModeStd(y_sd / 2, y_sd * 2))
    y_sigma_sd = numpyro.sample(
        'y_sigma_sd', dist_utils.gammaDistFromModeStd(y_sd / 2, y_sd * 2))
    y_sigma = numpyro.sample(
        'y_sigma', dist_utils.gammaDistFromModeStd(y_sigma_mode, y_sigma_sd).expand((nb_groups, )))

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        mean = a0 + a_grp[grp[idx]]
        numpyro.sample(
            'y', dist.StudentT(nu, mean, y_sigma[grp[idx]]), obs=y[idx])

    # Convert back to `b` to impose sum-to-zero constraint.
    a_grp_mean = jnp.mean(a_grp)
    numpyro.deterministic('b0', a0 + a_grp_mean)
    numpyro.deterministic('b_grp', a_grp - a_grp_mean)


def multi_nominal_predictors(y: jnp.ndarray, grp: jnp.ndarray, nb_groups: 'list[int]'):
    """
    Bayesian model as described in Chapter 20, Section 20.2, Figure 20.2

    Parameters
    ----------
    y: jnp.ndarray
        Metric predicted variable.
    grp: jnp.ndarray
        Nominal predictors.
    nb_groups: list[int]
        List of the number of unique groups in each column of `grp`.
    """
    assert y.shape[0] == grp.shape[0]
    assert y.ndim == 1 and grp.ndim == 2
    assert grp.shape[1] == len(nb_groups) == 2

    nb_obs = y.shape[0]

    # Predicted statistics.
    y_mean = jnp.mean(y)
    y_sd = jnp.std(y)

    # Priors for the intercept.
    a0_ = numpyro.sample('a0_', dist.Normal(0, 1))
    a0 = numpyro.deterministic('a0', a0_ * y_sd * 5 + y_mean)

    # Priors for coefficients associated with the first factor.
    a1_sigma = numpyro.sample(
        'a1_sigma', dist_utils.gammaDistFromModeStd(y_sd / 2, y_sd * 2))
    a1_ = numpyro.sample(
        'a1_', dist.Normal(0, 1).expand((nb_groups[0], )))
    a1 = numpyro.deterministic('a1', a1_ * a1_sigma)

    # Priors for coefficients associated with the second factor.
    a2_sigma = numpyro.sample(
        'a2_sigma', dist_utils.gammaDistFromModeStd(y_sd / 2, y_sd * 2))
    a2_ = numpyro.sample(
        'a2_', dist.Normal(0, 1).expand((nb_groups[1], )))
    a2 = numpyro.deterministic('a2', a2_ * a2_sigma)

    # Priors for coefficients associated with the interaction
    # between the first and the second factor.
    a1a2_sigma = numpyro.sample(
        'a1a2_sigma', dist_utils.gammaDistFromModeStd(y_sd / 2, y_sd * 2))
    a1a2_ = numpyro.sample(
        'a1a2_', dist.Normal(0, 1).expand(tuple(nb_groups)))
    a1a2 = numpyro.deterministic('a1a2', a1a2_ * a1a2_sigma)

    # Priors for y_sigma.
    y_sigma = numpyro.sample('y_sigma', dist.Uniform(y_sd / 100, y_sd * 10))

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        g1 = grp[idx, 0]
        g2 = grp[idx, 1]
        mean = a0 + a1[g1] + a2[g2] + a1a2[g1, g2]
        numpyro.sample('y', dist.Normal(mean, y_sigma), obs=y[idx])

    # Convert back to b to impose sum-to-zero constraint.
    m = a0 + a1[..., None] + a2[None, ...] + a1a2
    b0 = numpyro.deterministic('b0', jnp.mean(m))
    b1 = numpyro.deterministic('b1', jnp.mean(m, axis=1) - b0)
    b2 = numpyro.deterministic('b2', jnp.mean(m, axis=0) - b0)
    numpyro.deterministic('b1b2', m - (b0 + b1[..., None] + b2[None, ...]))


def multi_nominal_predictors_het_var_robust(y: jnp.ndarray, grp: jnp.ndarray, y_sds: jnp.ndarray, nb_groups: 'list[int]'):
    """
    Bayesian model as described in Chapter 20, Section 20.2, Figure 20.2

    Parameters
    ----------
    y: jnp.ndarray
        Metric predicted variable.
    grp: jnp.ndarray
        Nominal predictors.
    y_sds: jnp.ndarray
        Standard deviation of metric predicted variable `y` grouped by `grp` nominal variables.
    nb_groups: list[int]
        List of the number of unique groups in each column of `grp`.
    """
    assert y.shape[0] == grp.shape[0]
    assert y.ndim == 1 and grp.ndim == 2
    assert grp.shape[1] == len(nb_groups) == 2

    nb_obs = y.shape[0]

    # Predicted statistics.
    y_mean = jnp.mean(y)
    y_sd = jnp.std(y)

    y_sds_median = jnp.median(y_sds)
    y_sds_sd = jnp.std(y_sds)

    # Priors for the intercept.
    a0_ = numpyro.sample('a0_', dist.Normal(0, 1))
    a0 = numpyro.deterministic('a0', a0_ * y_sd * 5 + y_mean)

    # Priors for coefficients associated with the first factor.
    a1_sigma = numpyro.sample(
        'a1_sigma', dist_utils.gammaDistFromModeStd(y_sd / 2, y_sd * 2))
    a1_ = numpyro.sample(
        'a1_', dist.Normal(0, 1).expand((nb_groups[0], )))
    a1 = numpyro.deterministic('a1', a1_ * a1_sigma)

    # Priors for coefficients associated with the second factor.
    a2_sigma = numpyro.sample(
        'a2_sigma', dist_utils.gammaDistFromModeStd(y_sd / 2, y_sd * 2))
    a2_ = numpyro.sample(
        'a2_', dist.Normal(0, 1).expand((nb_groups[1], )))
    a2 = numpyro.deterministic('a2', a2_ * a2_sigma)

    # Priors for coefficients associated with the interaction
    # between the first and the second factor.
    a1a2_sigma = numpyro.sample(
        'a1a2_sigma', dist_utils.gammaDistFromModeStd(y_sd / 2, y_sd * 2))
    a1a2_ = numpyro.sample(
        'a1a2_', dist.Normal(0, 1).expand(tuple(nb_groups)))
    a1a2 = numpyro.deterministic('a1a2', a1a2_ * a1a2_sigma)

    # Priors for y_sigma.
    # y_sigma_mode = numpyro.sample(
    #     'y_sigma_mode', dist_utils.gammaDistFromModeStd(y_sds_median, y_sds_sd * 2))
    # y_sigma_sd = numpyro.sample(
    #     'y_sigma_sd', dist_utils.gammaDistFromModeStd(y_sds_median, y_sds_sd * 2))
    # y_sigma = numpyro.sample(
    #     'y_sigma',
    #     dist_utils.gammaDistFromModeStd(y_sigma_mode, y_sigma_sd).expand((nb_groups[1], )))
    # y_sigma_ = numpyro.sample(
    #     'y_sigma_',
    #     dist_utils.gammaDistFromModeStd(y_sds_median, y_sds_sd * 2).expand(tuple(nb_groups)))
    y_sigma_ = numpyro.sample(
        'y_sigma_',
        dist.Exponential(1. / y_sds_median).expand(tuple(nb_groups))
    )
    y_sigma = numpyro.deterministic(
        'y_sigma', jnp.maximum(y_sigma_, y_sds_median / 1000))
    # y_sigma = numpyro.sample('y_sigma', dist.Uniform(y_sd / 100, y_sd * 10))

    # y_sigma_mode = numpyro.sample(
    #     'y_sigma_mode', dist.LeftTruncatedDistribution(dist.Normal(y_sds_median, y_sds_sd * 2), low=0))
    # y_sigma_sd = numpyro.sample(
    #     'y_sigma_sd', dist.LeftTruncatedDistribution(dist.Normal(y_sds_median, y_sds_sd * 2), low=0))
    # y_sigma = numpyro.sample(
    #     'y_sigma',
    #     dist.LeftTruncatedDistribution(
    #         dist.Normal(y_sigma_mode, y_sigma_sd), low=0).expand(tuple(nb_groups))
    # )

    # Normality parameter.
    nu = numpyro.sample('nu', dist.Exponential(1. / 30))

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        g1 = grp[idx, 0]
        g2 = grp[idx, 1]
        mean = a0 + a1[g1] + a2[g2] + a1a2[g1, g2]
        numpyro.sample(
            'y', dist.StudentT(nu, mean, y_sigma[g1, g2]), obs=y[idx])

    # Convert back to b to impose sum-to-zero constraint.
    m = a0 + a1[..., None] + a2[None, ...] + a1a2
    b0 = numpyro.deterministic('b0', jnp.mean(m))
    b1 = numpyro.deterministic('b1', jnp.mean(m, axis=1) - b0)
    b2 = numpyro.deterministic('b2', jnp.mean(m, axis=0) - b0)
    numpyro.deterministic('b1b2', m - (b0 + b1[..., None] + b2[None, ...]))
