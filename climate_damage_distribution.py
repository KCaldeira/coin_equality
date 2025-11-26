"""
Income-rank-dependent climate damage distribution.

This module implements climate damage that varies by income level, with
lower-income populations experiencing proportionally greater losses.
This captures both aggregate damage effects and impacts on inequality.

Mathematical Foundation
-----------------------
For a Pareto income distribution with Gini index G and corresponding parameter a,
this module computes:

1. Aggregate damage Ω: fraction of total GDP lost to climate damage
2. Post-damage Gini G_climate: inequality after climate damage is applied

import numpy as np
from scipy.special import gammainc, gamma  # regularized & complete gamma
"""


def damage_integral(F0, F1, c_mean, gini, kc):
    """
    Compute ∫_{F0}^{F1} d(c(F)) dF, where:
    
      c(F) = c_mean * d/dF [ 1 - (1 - F)^(1 - 1/a) ]
      d(c) = kd * c * (1 - exp(-kc * c))
    
    and 0 <= F0 < F1 <= 1.
    
    Parameters
    ----------
    F0, F1 : float
        Integration limits in [0,1], with F0 < F1.
    c_mean : float
        Mean consumption parameter used in c(F).
    a : float
        Pareto/Lorenz shape parameter (a != 1).
    kd, kc : float
        Damage function parameters in d(c) = kd * c * exp(-kc * c).
    
    Returns
    -------
    float
        Value of the integral ∫_{F0}^{F1} d(c(F)) dF.
    """
    a = a_from_G(gini)

    if not (0.0 <= F0 < F1 <= 1.0):
        raise ValueError("Require 0 <= F0 < F1 <= 1.")
    if np.isclose(a, 1.0):
        raise ValueError("This closed-form expression assumes a != 1.")

    # A = c_mean * (1 - 1/a)
    A = c_mean * (1.0 - 1.0 / a)

    # z0, z1 as defined from the change of variables
    z0 = kc * A * (1.0 - F0) ** (-1.0 / a)
    z1 = kc * A * (1.0 - F1) ** (-1.0 / a)

    # Common front factor: kd * a * A * (kc * A)^(a - 1)
    front = a * A * (kc * A) ** (a - 1.0)

    # Second piece: γ(1-a, z1) - γ(1-a, z0)
    s = 1.0 - a

    #lower incomplete gamma terms
    #  Lower incomplete gamma: γ(s, x) = ∫_0^x t^{s-1} e^{-t} dt
    term_gamma =  (gammainc(s, z1) -gammainc(s,z0)) * gamma(s)

    # Combine
    integral_value = front *  term_gamma

    return integral_value

def calculate_climate_damage_ratio_from_prev_distribution(delta_T, prev_income_dist, params):
    """
    Calculate climate damage using income distribution from previous time step.

    This function avoids circular dependency by using the income distribution from the
    end of the previous time step to calculate damage in the current time step. This is
    physically reasonable because climate damage depends on the vulnerability of the
    population at the start of the period.

    Parameters
    ----------
    delta_T : float
        Temperature change above baseline (°C)
    prev_income_dist : dict
        Income distribution from previous time step:
        - 'y_mean': Mean per-capita income ($)
        - 'gini': Gini coefficient
    params : dict
        Model parameters including:
        - 'income_dependent_aggregate_damage': bool
        - 'income_dependent_damage_distribution': bool
        - 'y_damage_aggregate_scale': income scale for aggregate damage ($)
        - 'y_damage_distribution_scale': income scale for damage distribution ($)

    Returns
    -------
    Omega : float
        Aggregate damage fraction (0 ≤ Ω < 1)
    Gini_climate : float
        Post-damage Gini index (computed using previous distribution as base)
    """
    y_mean_prev = prev_income_dist['y_mean']
    gini_prev = prev_income_dist['gini']
    a = a_from_G(gini_prev)

    if income_dependent_tax_policy:
        Fcrit_tax = prev_income_dist['Fcrit_tax']
        income_ratio_Fcrit_tax = (1.0 - 1.0/a)* (1.0 - Fcrit_tax)**(-1.0/a)
        fract_damage_Fcrit_tax = income_ratio_Fcrit_tax * (1.0 - Fcrit_tax) * np.exp(-y_mean_prev * income_ratio_Fcrit_tax/y_damage_distribution_scale * y_mean_prev)

    else:
        Fcrit_tax = 1.0

    if income_dependent_redistribution_policy:
        Fcrit_redistribution = prev_income_dist['Fcrit_redistribution']
        income_ratio_Fcrit_redistribution = (1.0 - 1.0/a)* (1.0 - Fcrit_redistribution)**(-1.0/a)
        fract_damage_Fcrit_redistribution = income_ratio_Fcrit_redistribution * Fcrit_redistribution *  np.exp(-y_mean_prev * income_ratio_Fcrit_redistribution/y_damage_distribution_scale * y_mean_prev)
    else:
        Fcrit_redistribution = 0.0
        fract_damage_Fcrit_redistribution = 0.0

    middle_part = damage_integral(Fcrit_redistribution, Fcrit_tax, y_mean_prev, gini_prev, 1.0/y_damage_distribution_scale)
    
    damage_ratio = middle_part + fract_damage_Fcrit_tax - fract_damage_Fcrit_redistribution

    # Delegate to existing function with previous time step's distribution
    return damage_ratio 


def calculate_climate_damage_and_gini_effect(delta_T, Gini_current, y_mean, params):
    """
    Calculate climate damage and its effect on inequality.

    Supports two independent policy switches:
    - income_dependent_aggregate_damage: If True, aggregate damage decreases with wealth
    - income_dependent_damage_distribution: If True, damage is regressive (poor suffer more)

    Parameters
    ----------
    delta_T : float
        Temperature change above baseline (°C)
    Gini_current : float
        Current (pre-damage) Gini index
    y_mean : float
        Mean per-capita income ($)
    params : dict
        - 'psi1': linear damage coefficient (°C⁻¹)
        - 'psi2': quadratic damage coefficient (°C⁻²)
        - 'income_dependent_aggregate_damage': bool
        - 'income_dependent_damage_distribution': bool
        - 'y_damage_aggregate_scale': income scale for aggregate damage ($)
        - 'y_damage_distribution_scale': income scale for damage distribution ($)

    Returns
    -------
    Omega : float
        Aggregate damage fraction (0 ≤ Ω < 1)
    Gini_climate : float
        Post-damage Gini index
    """
    if delta_T <= 0:
        return 0.0, Gini_current

    psi1 = params['psi1']
    psi2 = params['psi2']
    income_dependent_aggregate = params['income_dependent_aggregate_damage']
    income_dependent_distribution = params['income_dependent_damage_distribution']

    # Base damage from temperature (DICE-like formula)
    omega_base = psi1 * delta_T + psi2 * (delta_T ** 2)
    omega_base = min(omega_base, 1.0 - EPSILON)

    # === Aggregate damage calculation ===
    if income_dependent_aggregate and Gini_current > 0 and Gini_current < 1:
        # Wealthier societies experience less aggregate damage
        a = a_from_G(Gini_current)
        y_aggregate_scale = params['y_damage_aggregate_scale']
        income_damage_scale = pareto_integral_scipy(y_mean, a, y_aggregate_scale)
        omega_aggregate = omega_base * income_damage_scale
    else:
        # DICE-like: damage independent of income
        omega_aggregate = omega_base

    # === Damage distribution calculation ===
    if not income_dependent_distribution:
        # Uniform damage distribution: everyone loses the same fraction
        # No effect on Gini
        return float(omega_aggregate), float(Gini_current)

    # Income-dependent distribution: poor suffer more
    if Gini_current <= 0 or Gini_current >= 1:
        # Edge case: no inequality or invalid Gini
        return float(omega_aggregate), float(Gini_current)

    y_dist_scale = params['y_damage_distribution_scale']

    # Uniform damage special case: very large scale means effectively uniform
    if y_dist_scale > INVERSE_EPSILON:
        return float(omega_aggregate), float(Gini_current)

    # Convert Gini → Pareto parameter (if not already computed)
    if not income_dependent_aggregate or Gini_current <= 0 or Gini_current >= 1:
        a = a_from_G(Gini_current)
    lorenz_exponent = 1.0 - 1.0 / a

    # Dimensionless parameter for regressivity
    b = y_dist_scale / (y_mean * lorenz_exponent)

    # === Regressive damage distribution ===
    # Closed form using hypergeometric functions
    H1 = hyp2f1(1.0, a, a + 1.0, -b)  # Mean damage factor
    H2 = hyp2f1(1.0, 2.0 * a, 2.0 * a + 1.0, -b)  # Inequality adjustment

    omega_scaled = omega_aggregate * (y_dist_scale / y_mean)
    Omega = omega_scaled * H1

    # === Post-damage Gini (G_climate) ===
    Gini_climate = (Gini_current + omega_scaled * (H2 - H1)) / (1.0 - omega_scaled * H1)
    if np.isnan(Gini_climate) or Gini_climate < 0.0:
        print("Warning: Gini_climate computation produced invalid value. Setting to 0.0.")
        print(f"  Inputs: delta_T={delta_T}, Gini_current={Gini_current}, y_mean={y_mean}, params={params}")
        print(f"  Computed: Omega={Omega}, Gini_climate (raw)={Gini_climate}")
        Gini_climate = 0.0

    return float(Omega), float(Gini_climate)

