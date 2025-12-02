"""
Analytical integration functions for utility calculations over income distributions.

This module provides closed-form solutions for integrating CRRA utility functions
over Pareto income distributions with various modifications (taxation, redistribution, etc.).
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1
from constants import LOOSE_EPSILON

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


def crra_utility_integral(F0, F1, C, a, eta, k, tol=LOOSE_EPSILON):
    """
    Integral of CRRA utility over rank F in [F0, F1], with

        c(F) = C * (1 - F)**(-1/a) + k

    and CRRA utility

        u(c) = (c**(1 - eta)) / (1 - eta),   eta != 1
        u(c) = log(c),                       eta == 1

    Parameters
    ----------
    F0, F1 : float
        Lower and upper ranks (0 <= F0 < F1 <= 1).
    C : float
        Scale parameter in the Pareto part of consumption.
    a : float
        Pareto shape parameter (> 0).
    eta : float
        CRRA coefficient. eta = 1 is log utility.
    k : float, optional
        Additive constant in consumption. Default 0.
    tol : float, optional
        Tolerance for treating eta as 1 or other special cases (default LOOSE_EPSILON = 1e-8).

    Returns
    -------
    U : float
        Integral of utility over F in [F0, F1].

    Notes
    -----
    Uses analytical solutions based on hypergeometric functions when k != 0.
    Falls back to numerical integration only when necessary (log utility with k != 0).

    For the case k = 0 (pure Pareto distribution):
    - CRRA: Uses closed form with power functions
    - Log utility: Uses analytical integration of log(C(1-F)^(-1/a))

    For k != 0 (Pareto + additive constant, e.g., redistribution):
    - CRRA: Uses hypergeometric function 2F1
    - Log utility: Uses numerical integration
    """

    F0 = float(F0)
    F1 = float(F1)
    C = float(C)
    a = float(a)
    eta = float(eta)
    k = float(k)

    # --- helper: numeric integral for log utility (eta == 1) and general k ---
    def _numeric_log_integral(F0, F1, C, a, k):
        def integrand(F):
            c = C * (1.0 - F)**(-1.0 / a) + k
            if c <= 0:
                raise ValueError("Consumption must remain positive for log utility.")
            return np.log(c)
        val, _ = quad(integrand, F0, F1, limit=200)
        return val

    # --- log utility: eta == 1 ---
    if abs(eta - 1.0) < tol:
        # Special analytic case: k = 0
        if abs(k) < tol:
            # u(c) = log(C (1-F)^(-1/a)) = log C - (1/a) log(1-F)
            def I2(F):
                x = 1.0 - F
                if x <= 0:
                    raise ValueError("1 - F must be > 0 inside the log.")
                return -x * np.log(x) + x
            return ((F1 - F0) * np.log(C)
                    - (1.0 / a) * (I2(F1) - I2(F0)))
        else:
            # General k != 0: fall back on numeric integration
            return _numeric_log_integral(F0, F1, C, a, k)

    # --- CRRA case: eta != 1 ---
    beta = (1.0 - eta) / a

    # Special analytic case: k = 0
    if abs(k) < tol:
        # U(F0,F1) = C^{1-eta} / [(1-eta)(1 - beta)]
        #           * [(1-F0)^{1 - beta} - (1-F1)^{1 - beta}]
        # except when 1 - beta = 0 (then it's logarithmic in 1-F).
        if abs(1.0 - beta) < tol:
            # (1-eta)/a = 1 -> integrand ~ 1/(1-F)
            x0 = 1.0 - F0
            x1 = 1.0 - F1
            if x0 <= 0 or x1 <= 0:
                raise ValueError("1 - F must be > 0 inside the log.")
            return (C**(1.0 - eta) / (1.0 - eta)) * (np.log(x0) - np.log(x1))
        else:
            x0 = 1.0 - F0
            x1 = 1.0 - F1
            return (C**(1.0 - eta)
                    / ((1.0 - eta) * (1.0 - beta))
                    * (x0**(1.0 - beta) - x1**(1.0 - beta)))

    # General case: eta != 1 and k != 0
    # Use hypergeometric closed form:
    #
    # Let x = 1 - F. Then
    # U = C^{1-eta} / [(1-eta) * mu] *
    #     [ x^mu * 2F1(eta-1, mu/b; 1+mu/b; -(k/C)*x^b) ]_{x1}^{x0}
    #
    b = 1.0 / a
    mu = 1.0 - (1.0 - eta) / a
    q = k / C
    prefactor = C**(1.0 - eta) / ((1.0 - eta) * mu)

    def G(x):
        # x > 0 in (0,1]
        return x**mu * hyp2f1(eta - 1.0, mu / b, 1.0 + mu / b, -q * x**b)

    x0 = 1.0 - F0
    x1 = 1.0 - F1
    if x0 <= 0 or x1 <= 0:
        raise ValueError("1 - F must be > 0 in the support of the integral.")

    return float(prefactor * (G(x0) - G(x1)))


def crra_utility_integral_with_damage(
    F0,
    F1,
    Fmin,
    Fmax_for_clip,
    y_mean_before_damage,
    omega_base,
    y_damage_distribution_scale,
    uniform_redistribution,
    gini,
    eta,
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
    y_damage_distribution_scale : float
        Damage distribution scale parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    eta : float
        CRRA coefficient (eta = 1 is log utility, eta != 1 is power utility).
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
    - Climate damage (income-dependent via Lambert W)
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
    income_vals = y_of_F_after_damage(
        F_nodes,
        Fmin,
        Fmax_for_clip,
        y_mean_before_damage,
        omega_base,
        y_damage_distribution_scale,
        uniform_redistribution,
        gini,
        branch=branch,
    )

    # Compute utility at each node
    if abs(eta - 1.0) < LOOSE_EPSILON:
        # Log utility
        if np.any(income_vals <= 0):
            raise ValueError("Income must be positive for log utility")
        utility_vals = np.log(income_vals)
    else:
        # Power utility
        if np.any(income_vals < 0):
            raise ValueError("Income must be non-negative for CRRA utility")
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
    y_damage_distribution_scale,
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
    y_damage_distribution_scale : float
        Damage distribution scale parameter.
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
        damage(F) = omega_base * exp(-income(F) / y_damage_distribution_scale)

    Income at each rank is computed using y_of_F_after_damage() which accounts for:
    - Pareto-Lorenz income distribution
    - Climate damage (income-dependent via Lambert W)
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
        y_damage_distribution_scale,
        uniform_redistribution,
        gini,
        branch=branch,
    )

    # Compute damage at each node
    # damage(F) = omega_base * exp(-income(F) / y_damage_distribution_scale)
    damage_vals = omega_base * np.exp(-income_vals / y_damage_distribution_scale)

    # Weighted sum scaled by interval length
    integral = F_half * np.dot(wi, damage_vals)

    return float(integral)
