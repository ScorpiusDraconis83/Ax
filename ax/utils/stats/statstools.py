#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from logging import Logger
from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from ax.core.data import Data
from ax.utils.common.logger import get_logger

logger: Logger = get_logger(__name__)
# pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
num_mixed = Union[np.ndarray, list[float]]


def inverse_variance_weight(
    means: npt.NDArray,
    variances: npt.NDArray,
    conflicting_noiseless: str = "warn",
) -> tuple[float, float]:
    """Perform inverse variance weighting.

    Args:
        means: The means of the observations.
        variances: The variances of the observations.
        conflicting_noiseless: How to handle the case of
            multiple observations with zero variance but different means.
            Options are "warn" (default), "ignore" or "raise".

    """
    if conflicting_noiseless not in {"warn", "ignore", "raise"}:
        raise ValueError(
            f"Unsupported option `{conflicting_noiseless}` for conflicting_noiseless."
        )
    if len(means) != len(variances):
        raise ValueError("Means and variances must be of the same length.")
    # new_mean = \sum_i 1/var_i mean_i / \sum_i (1/var_i), unless any var = 0,
    # in which case we report the mean of all values with var = 0.
    idx_zero = variances == 0
    if idx_zero.any():
        means_z = means[idx_zero]
        if np.var(means_z) > 0:
            message = "Multiple observations zero variance but different means."
            if conflicting_noiseless == "warn":
                logger.warning(message)
            elif conflicting_noiseless == "raise":
                raise ValueError(message)
        return np.mean(means_z), 0
    inv_vars = np.divide(1.0, variances)
    sum_inv_vars = inv_vars.sum()
    new_mean = np.inner(inv_vars, means) / sum_inv_vars
    new_var = np.divide(1.0, sum_inv_vars)
    return new_mean, new_var


def total_variance(
    means: npt.NDArray,
    variances: npt.NDArray,
    sample_sizes: npt.NDArray,
) -> float:
    """Compute total variance."""
    variances = variances * sample_sizes
    weighted_variance_of_means = np.average(
        (means - means.mean()) ** 2, weights=sample_sizes
    )
    weighted_mean_of_variance = np.average(variances, weights=sample_sizes)
    return (weighted_variance_of_means + weighted_mean_of_variance) / sample_sizes.sum()


def positive_part_james_stein(
    means: num_mixed,
    sems: num_mixed,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Estimation method for Positive-part James-Stein estimator.

    This method takes a vector of K means (`y_i`) and standard errors
    (`sigma_i`) and calculates the positive-part James Stein estimator.

    Resulting estimates are the shrunk means and standard errors. The positive
    part James-Stein estimator shrinks each constituent average to the grand
    average:

        y_i - phi_i * y_i + phi_i * ybar

    The variable phi_i determines the amount of shrinkage. For `phi_i = 1`,
    `mu_hat` is equal to `ybar` (the mean of all `y_i`), while for `phi_i = 0`,
    `mu_hat` is equal to `y_i`. It can be shown that restricting `phi_i <= 1`
    dominates the unrestricted estimator, so this method restricts `phi_i` in
    this manner. The amount of shrinkage, `phi_i`, is determined by:

        (K - 3) * sigma2_i / s2

    That is, less shrinkage is applied when individual means are estimated with
    greater precision, and more shrinkage is applied when individual means are
    very tightly clustered together. We also restrict `phi_i` to never be larger
    than 1.

    The variance of the mean estimator is:

        (1 - phi_i) * sigma2_i
        + phi * sigma2_i / K
        + 2 * phi_i ** 2 * (y_i - ybar)^2 / (K - 3)

    The first term is the variance component from `y_i`, the second term is the
    contribution from the mean of all `y_i`, and the third term is the
    contribution from the uncertainty in the sum of squared deviations of `y_i`
    from the mean of all `y_i`.

    For more information, see
    https://ax.dev/docs/models.html#empirical-bayes-and-thompson-sampling.

    Args:
        means: Means of each arm
        sems: Standard errors of each arm
    Returns:
        mu_hat_i: Empirical Bayes estimate of each arm's mean
        sem_i: Empirical Bayes estimate of each arm's sem
    """
    if np.min(sems) < 0:
        raise ValueError("sems cannot be negative.")
    y_i = np.array(means)
    K = y_i.shape[0]
    if K < 4:
        raise ValueError(
            "Less than 4 measurements passed to positive_part_james_stein. "
            + "Returning raw estimates."
        )
    sigma2_i = np.power(sems, 2)
    ybar = np.mean(y_i)
    s2 = np.var(y_i - ybar, ddof=3)  # sample variance normalized by K-3
    phi_i = np.ones_like(sigma2_i) if s2 == 0 else np.minimum(1, sigma2_i / s2)
    mu_hat_i = y_i + phi_i * np.subtract(ybar, y_i)

    sigma_hat_i = np.sqrt(
        np.subtract(1.0, phi_i) * sigma2_i
        + phi_i * sigma2_i / K
        + np.multiply(2, phi_i**2) * (y_i - ybar) ** 2 / (K - 3)
    )
    return mu_hat_i, sigma_hat_i


def relativize(
    means_t: npt.NDArray | list[float] | float,
    sems_t: npt.NDArray | list[float] | float,
    mean_c: npt.NDArray | float,
    sem_c: npt.NDArray | float,
    bias_correction: bool = True,
    cov_means: npt.NDArray | list[float] | float = 0.0,
    as_percent: bool = False,
    control_as_constant: bool = False,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Ratio estimator based on the delta method.

    This uses the delta method (i.e. a Taylor series approximation) to estimate
    the mean and standard deviation of the sampling distribution of the ratio
    between test and control -- that is, the sampling distribution of an
    estimator of the true population value under the assumption that the means
    in test and control have a known covariance:

        (mu_t / mu_c) - 1.

    Under a second-order Taylor expansion, the sampling distribution of the
    relative change in empirical means, which is `m_t / m_c - 1`, is
    approximately normally distributed with mean

        [(mu_t - mu_c) / mu_c] - [(sigma_c)^2 * mu_t] / (mu_c)^3

    and variance

        (sigma_t / mu_c)^2
        - 2 * mu_t _ sigma_tc / mu_c^3
        + [(sigma_c * mu_t)^2 / (mu_c)^4]

    as the higher terms are assumed to be close to zero in the full Taylor
    series. To estimate these parameters, we plug in the empirical means and
    standard errors. This gives us the estimators:

        [(m_t - m_c) / m_c] - [(s_c)^2 * m_t] / (m_c)^3

    and

        (s_t / m_c)^2 - 2 * m_t * s_tc / m_c^3 + [(s_c * m_t)^2 / (m_c)^4]

    Note that the delta method does NOT take as input the empirical standard
    deviation of a metric, but rather the standard error of the mean of that
    metric -- that is, the standard deviation of the metric after division by
    the square root of the total number of observations.

    Args:
        means_t: Sample means (test)
        sems_t: Sample standard errors of the means (test)
        mean_c: Sample mean (control)
        sem_c: Sample standard error of the mean (control)
        bias_correction: Whether to apply bias correction when computing relativized
            metric values. Uses a second-order Taylor expansion for approximating
            the means and standard errors of the ratios.
        cov_means: Sample covariance between test and control
        as_percent: If true, return results in percent (* 100)
        control_as_constant: If true, control is treated as a constant.
            bias_correction, sem_c, and cov_means are ignored when this is true.


    Returns:
        rel_hat: Inferred means of the sampling distribution of
            the relative change `(mean_t - mean_c) / abs(mean_c)`
        sem_hat: Inferred standard deviation of the sampling
            distribution of rel_hat -- i.e. the standard error.

    """
    # if mean_c is too small, bail
    epsilon = 1e-10
    if np.any(np.abs(mean_c) < epsilon):
        raise ValueError(
            "mean_control ({} +/- {}) is smaller than 1 in 10 billion, "
            "which is too small to reliably analyze ratios using the delta "
            "method. This usually occurs because winsorization has truncated "
            "all values down to zero. Try using a delta type that applies "
            "no winsorization.".format(mean_c, sem_c)
        )
    m_t = np.array(means_t)
    s_t = np.array(sems_t)
    cov_t = np.array(cov_means)
    abs_mean_c = np.abs(mean_c)
    r_hat = (m_t - mean_c) / abs_mean_c

    if control_as_constant:
        var = (s_t / abs_mean_c) ** 2
    else:
        c = m_t / mean_c
        if bias_correction:
            r_hat = r_hat - m_t * sem_c**2 / abs_mean_c**3

        # If everything's the same, then set r_hat to zero
        same = (m_t == mean_c) & (s_t == sem_c)
        r_hat = ~same * r_hat
        var = ((s_t**2) - 2 * c * cov_t + (c**2) * (sem_c**2)) / (mean_c**2)
    if as_percent:
        return (r_hat * 100, np.sqrt(var) * 100)
    else:
        return (r_hat, np.sqrt(var))


def unrelativize(
    means_t: npt.NDArray | list[float] | float,
    sems_t: npt.NDArray | list[float] | float,
    mean_c: float,
    sem_c: float,
    bias_correction: bool = True,
    cov_means: npt.NDArray | list[float] | float = 0.0,
    as_percent: bool = False,
    control_as_constant: bool = False,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Reverse operation of ax.utils.stats.statstools.relativize.

    Args:
        means_t: Relativized sample means (test) to be unrelativized
        sems_t: Relativized sample SEM of the means (test) to be unrelativized
        mean_c: Unrelativized control mean
        sem_c: Unrelativized control SEM of the mean
        bias_correction: if `means_t` and `sems_t` are obtained with
                         `bias_correction=True` in ax.utils.stats.statstools.relativize
        cov_means: Sample covariance between the **unrelativized** test and control
        as_percent: If true, assuming `means_t` and `sems_t` are percentages
                    (i.e., 1 means 1%).
        control_as_constant: If true, control is treated as a constant.
            bias_correction, sem_c, and cov_means are ignored when this is true.

    Returns:
        m_t: Inferred sample (test) means in the unrelativized scale
        s_t: Inferred SEM of sample (test) means in the unrelativized scale
    """
    means_t = np.array(means_t, dtype=float)
    sems_t = np.array(sems_t, dtype=float)

    if as_percent:
        means_t = means_t / 100
        sems_t = sems_t / 100

    abs_mean_c = np.abs(mean_c)
    m_t = means_t * abs_mean_c + mean_c

    if control_as_constant:
        s_t = sems_t * abs_mean_c
    else:
        if bias_correction:
            m_t = m_t / (1 - (sem_c / abs_mean_c) ** 2)

        var = sems_t**2
        c = m_t / mean_c
        s_t2 = var * (mean_c**2) + 2 * c * cov_means - (c**2) * (sem_c**2)

        # This is only positive when sems_t > sem_c * mean_c * (means_t + 1)
        # If above condition cannot be guaranteed, use control_as_constant = True
        s_t = np.sqrt(s_t2.clip(min=0.0))

    # if means_t is 0.0 exactly, return control mean and sem directly
    if np.isscalar(means_t):
        if means_t == 0.0:
            m_t = mean_c
            s_t = sem_c
    else:
        m_t = np.array(m_t)
        s_t = np.array(s_t)
        m_t[means_t == 0.0] = mean_c
        s_t[means_t == 0.0] = sem_c

    # pyre-fixme[7]: Expected `Tuple[ndarray[typing.Any, typing.Any],
    #  ndarray[typing.Any, typing.Any]]` but got `Tuple[Union[ndarray[typing.Any,
    #  dtype[typing.Any]], float], Union[ndarray[typing.Any, dtype[typing.Any]],
    #  float]]`.
    return m_t, s_t


def agresti_coull_sem(
    n_numer: pd.Series | npt.NDArray | int,
    n_denom: pd.Series | npt.NDArray | int,
    prior_successes: int = 2,
    prior_failures: int = 2,
) -> npt.NDArray | float:
    """Compute the Agresti-Coull style standard error for a binomial proportion.

    Reference:
    *Agresti, Alan, and Brent A. Coull. Approximate Is Better than 'Exact' for
    Interval Estimation of Binomial Proportions." The American Statistician,
    vol. 52, no. 2, 1998, pp. 119-126. JSTOR, www.jstor.org/stable/2685469.*

    """
    n_numer = np.array(n_numer)
    n_denom = np.array(n_denom)
    p_for_sem = (n_numer + prior_successes) / (
        n_denom + prior_successes + prior_failures
    )
    sem = np.sqrt(p_for_sem * (1 - p_for_sem) / n_denom)
    return sem


def marginal_effects(
    df: pd.DataFrame, covariates: list[str] | None = None
) -> pd.DataFrame:
    """
    This method calculates the relative (in %) change in the outcome achieved
    by using any individual factor level versus randomizing across all factor
    levels. It does this by estimating a baseline under the experiment by
    marginalizing over all factors/levels. For each factor level, then,
    it conditions on that level for the individual factor and then marginalizes
    over all levels for all other factors.

    Args:
        df: Dataframe containing columns named mean and sem. All other columns
            are assumed to be factors for which to calculate marginal effects.
        covariates: List of columns to be used as covariates. If None, then use
            all columns in df that are not named "mean" or "sem".

    Returns:
        A dataframe containing columns "Name", "Level", "Beta" and "SE"
            corresponding to the factor, level, effect and standard error.
            Results are relativized as percentage changes.
    """
    covariates = covariates or [col for col in df.columns if col not in ["mean", "sem"]]
    formatted_vals = []
    overall_mean, overall_sem = inverse_variance_weight(
        df["mean"],
        np.power(df["sem"], 2),
    )
    for cov in covariates:
        if len(df[cov].unique()) <= 1:
            next
        df_gb = df.groupby(cov)
        for name, group_df in df_gb:
            group_mean, group_var = inverse_variance_weight(
                group_df["mean"], np.power(group_df["sem"], 2)
            )
            effect, effect_sem = relativize(
                group_mean,
                np.sqrt(group_var),
                overall_mean,
                overall_sem,
                cov_means=0.0,
                as_percent=True,
            )
            formatted_vals.append(
                {"Name": cov, "Level": name, "Beta": effect, "SE": effect_sem}
            )
    return pd.DataFrame(formatted_vals)[["Name", "Level", "Beta", "SE"]]


def relativize_data(
    data: Data,
    status_quo_name: str = "status_quo",
    as_percent: bool = False,
    include_sq: bool = False,
    bias_correction: bool = True,
    control_as_constant: bool = False,
) -> Data:
    """Relativize a data object w.r.t. a status_quo arm.

    Args:
        data: The data object to be relativized.
        status_quo_name: The name of the status_quo arm.
        as_percent: If True, return results as percentage change.
        include_sq: Include status quo in final df.
        bias_correction: Whether to apply bias correction when computing relativized
            metric values. Uses a second-order Taylor expansion for approximating
            the means and standard errors or the ratios, see
            ax.utils.stats.statstools.relativize for more details.
        control_as_constant: If true, control is treated as a constant.
            bias_correction is ignored when this is true.

    Returns:
        The new data object with the relativized metrics (excluding the
            status_quo arm)

    """
    df = data.df.copy()
    grp_cols = list(
        {"trial_index", "metric_name", "random_split"}.intersection(df.columns.values)
    )

    grouped_df = df.groupby(grp_cols)
    dfs = []
    for grp in grouped_df.groups.keys():
        subgroup_df = grouped_df.get_group(grp)
        is_sq = subgroup_df["arm_name"] == status_quo_name
        sq_mean, sq_sem = (
            subgroup_df[is_sq][["mean", "sem"]].drop_duplicates().values.flatten()
        )

        # rm status quo from final df to relativize
        if not include_sq:
            subgroup_df = subgroup_df[~is_sq]
        means_rel, sems_rel = relativize(
            means_t=subgroup_df["mean"].values,
            sems_t=subgroup_df["sem"].values,
            mean_c=sq_mean,
            sem_c=sq_sem,
            as_percent=as_percent,
            bias_correction=bias_correction,
            control_as_constant=control_as_constant,
        )
        dfs.append(
            pd.concat(
                [
                    subgroup_df.drop(["mean", "sem"], axis=1),
                    pd.DataFrame(
                        np.array([means_rel, sems_rel]).T,
                        columns=["mean", "sem"],
                        index=subgroup_df.index,
                    ),
                ],
                axis=1,
            )
        )
    df_rel = pd.concat(dfs, axis=0)
    if include_sq:
        df_rel.loc[df_rel["arm_name"] == status_quo_name, "sem"] = 0.0
    return Data(df_rel)
