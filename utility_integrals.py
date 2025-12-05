"""
Analytical integration functions for utility calculations over income distributions.

This module provides closed-form solutions for integrating CRRA utility functions
over Pareto income distributions with various modifications (taxation, redistribution, etc.).
"""

import numpy as np
from constants import LOOSE_EPSILON, EPSILON

def crra_utility_interval(F0, F1, c_mean, eta):
    """
    Utility of a constant consumption level c over the interval [F0, F1].

    u(c) = c^(1-eta)/(1-eta)    if eta != 1
           ln(c)               if eta == 1
    """
    width = F1 - F0
    if width < 0 or F0 < 0 or F1 > 1:
        raise ValueError("Require 0 <= F0 <= F1 <= 1.")

    if eta == 1:
        return width * np.log(c_mean)
    else:
        return width * (c_mean**(1-eta)) / (1-eta)


def crra_utility_integral_with_damage(
    F0,
    F1,
    Fmin,
    Fmax_for_clip,
    y_mean_before_damage,
    omega_base,
    y_damage_distribution_exponent,
    y_net_reference,
    uniform_redistribution,
    gini,
    eta,
    s,
    xi,
    wi,
    branch=0,
):
    """
    Numerically integrate CRRA utility over rank F in [F0, F1] accounting for climate damage.

    Uses y_of_F_after_damage() to compute income at each rank, then integrates utility
    using Gauss-Legendre quadrature.

    Parameters
    ----------
    F0 : float
        Lower rank (0 <= F0 < F1 <= 1).
    F1 : float
        Upper rank (0 <= F0 < F1 <= 1).
    Fmin : float
        Minimum rank for clipping in y_of_F_after_damage.
    Fmax_for_clip : float
        Maximum rank for clipping in y_of_F_after_damage.
    y_mean_before_damage : float
        Mean income before damage.
    omega_base : float
        Base climate damage parameter.
    y_damage_distribution_exponent : float
        Damage distribution coefficient parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    eta : float
        CRRA coefficient (eta = 1 is log utility, eta != 1 is power utility).
    s : float
        Savings rate (0 <= s < 1). Income after savings is (1-s)*income.
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1].
    wi : ndarray
        Gauss-Legendre quadrature weights.
    branch : int, optional
        Lambert W branch (default 0).

    Returns
    -------
    U : float
        Integral of utility over F in [F0, F1].

    Notes
    -----
    CRRA utility:
        u(c) = c^(1-eta) / (1-eta)   if eta != 1
        u(c) = log(c)                if eta == 1

    Income at each rank is computed using y_of_F_after_damage() which accounts for:
    - Pareto-Lorenz income distribution
    - Climate damage (income-dependent via power-law)
    - Redistribution
    """
    from income_distribution import y_of_F_after_damage

    F0 = float(F0)
    F1 = float(F1)
    eta = float(eta)

    if F1 <= F0:
        raise ValueError(f"Require F0 < F1, got F0={F0}, F1={F1}")
    if F0 < 0 or F1 > 1:
        raise ValueError(f"Require 0 <= F0 < F1 <= 1, got F0={F0}, F1={F1}")

    # Map Gauss-Legendre nodes from [-1, 1] to [F0, F1]
    F_mid = 0.5 * (F1 + F0)
    F_half = 0.5 * (F1 - F0)
    F_nodes = F_half * xi + F_mid

    # Evaluate income at quadrature nodes (accounting for climate damage)
    income_before_savings = y_of_F_after_damage(
        F_nodes,
        Fmin,
        Fmax_for_clip,
        y_mean_before_damage,
        omega_base,
        y_damage_distribution_exponent,
        y_net_reference,
        uniform_redistribution,
        gini,
        branch=branch,
    )

    # Apply savings rate to get consumption
    income_vals = income_before_savings * (1.0 - s)

    # Limit income to EPSILON to prevent negative or zero values in utility calculation
    income_vals = np.maximum(income_vals, EPSILON)

    # Compute utility at each node
    if abs(eta - 1.0) < LOOSE_EPSILON:
        # Log utility
        utility_vals = np.log(income_vals)
    else:
        # Power utility
        utility_vals = (income_vals ** (1.0 - eta)) / (1.0 - eta)

    # Weighted sum scaled by interval length
    integral = F_half * np.dot(wi, utility_vals)

    return float(integral)


def climate_damage_integral(
    F0,
    F1,
    Fmin,
    Fmax_for_clip,
    y_mean_before_damage,
    omega_base,
    y_damage_distribution_exponent,
    y_net_reference,
    uniform_redistribution,
    gini,
    xi,
    wi,
    branch=0,
):
    """
    Numerically integrate climate damage over rank F in [F0, F1].

    Uses y_of_F_after_damage() to compute income at each rank, then integrates damage
    using Gauss-Legendre quadrature.

    Parameters
    ----------
    F0 : float
        Lower rank (0 <= F0 < F1 <= 1).
    F1 : float
        Upper rank (0 <= F0 < F1 <= 1).
    Fmin : float
        Minimum rank for clipping in y_of_F_after_damage.
    Fmax_for_clip : float
        Maximum rank for clipping in y_of_F_after_damage.
    y_mean_before_damage : float
        Mean income before damage.
    omega_base : float
        Base climate damage parameter.
    y_damage_distribution_exponent : float
        Damage distribution coefficient parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1].
    wi : ndarray
        Gauss-Legendre quadrature weights.
    branch : int, optional
        Lambert W branch (default 0).

    Returns
    -------
    D : float
        Integral of damage over F in [F0, F1].

    Notes
    -----
    Climate damage at each rank:
        damage(F) = omega_base * (income(F) / y_net_reference)**y_damage_distribution_exponent

    Income at each rank is computed using y_of_F_after_damage() which accounts for:
    - Pareto-Lorenz income distribution
    - Climate damage (income-dependent via power-law)
    - Redistribution
    """
    from income_distribution import y_of_F_after_damage

    F0 = float(F0)
    F1 = float(F1)

    if F1 <= F0:
        raise ValueError(f"Require F0 < F1, got F0={F0}, F1={F1}")
    if F0 < 0 or F1 > 1:
        raise ValueError(f"Require 0 <= F0 < F1 <= 1, got F0={F0}, F1={F1}")

    # Map Gauss-Legendre nodes from [-1, 1] to [F0, F1]
    F_mid = 0.5 * (F1 + F0)
    F_half = 0.5 * (F1 - F0)
    F_nodes = F_half * xi + F_mid

    # Evaluate income at quadrature nodes (accounting for climate damage)
    income_vals = y_of_F_after_damage(
        F_nodes,
        Fmin,
        Fmax_for_clip,
        y_mean_before_damage,
        omega_base,
        y_damage_distribution_exponent,
        y_net_reference,
        uniform_redistribution,
        gini,
        branch=branch,
    )

    # Compute damage at each node
    # damage(F) = omega_base * (income(F) / y_net_reference)**y_damage_distribution_exponent
    damage_vals = omega_base * (income_vals / y_net_reference)**y_damage_distribution_exponent

    # Weighted sum scaled by interval length
    integral = F_half * np.dot(wi, damage_vals)

    return float(integral)
