"""
Income-rank-dependent climate damage distribution.

This module implements climate damage that varies by income level, with
lower-income populations experiencing proportionally greater losses.
This captures both aggregate damage effects and impacts on inequality.
"""

import numpy as np
from income_distribution import a_from_G
from mpmath import hyp2f1


def calculate_climate_damage_and_gini_effect(delta_T, Gini_current, y_mean, params):
    """
    Calculate income-dependent climate damage and its effect on inequality.

    Uses a discretized approximation with population bins for computational efficiency.
    Climate damage is applied as a function of income level using a half-saturation
    (Michaelis-Menten) model, with lower incomes experiencing higher fractional losses.

    Parameters
    ----------
    delta_T : float
        Temperature change above baseline (°C)
    Gini_current : float
        Current Gini index before climate damage (0 < Gini < 1)
    y_mean : float
        Mean per-capita income ($)
    params : dict
        Must include:
        - 'k_damage_coeff': base damage coefficient (dimensionless)
        - 'k_damage_exp': temperature exponent (dimensionless)
        - 'k_damage_halfsat': income half-saturation constant ($)
          (income level at which damage is 50% of maximum)
    n_bins : int, optional
        Number of population bins for discretization (default: 1000)

    Returns
    -------
    Omega : float
        Aggregate damage fraction: fraction of total GDP lost to climate damage
        (0 ≤ Omega < 1)
    Gini_climate : float
        Gini index of income distribution after climate damage is applied
        (typically Gini_climate > Gini_current for finite k_damage_halfsat)

    Notes
    -----
    The damage function uses a half-saturation model:
        ω_max(ΔT) = k_damage_coeff · ΔT^k_damage_exp
        ω(y, ΔT) = ω_max · (1 - y / (k_damage_halfsat + y))

    where:
    - At income y = 0: ω = ω_max (maximum damage for poorest)
    - At income y = k_damage_halfsat: ω = ω_max/2 (half of maximum damage)
    - As income y → ∞: ω → 0 (damage approaches zero for wealthy)

    The aggregate damage fraction is the income-weighted average:
        Ω = Σ[ω(y(F)) · y(F)] / Σ[y(F)]

    The post-damage Gini is computed from the distribution:
        y_damaged(F) = (1 - ω(y(F))) · y(F)

    Special cases:
    - k_damage_halfsat → ∞: uniform damage, Omega = ω_max, Gini unchanged
    - k_damage_halfsat → 0: maximum regressive damage
    - ΔT = 0: no damage, Omega = 0, Gini_climate = Gini_current

    Examples
    --------
    Uniform damage (very high k_damage_halfsat):
    >>> params = {'k_damage_coeff': 0.02, 'k_damage_exp': 2.0,
    ...           'k_damage_halfsat': 1e12}
    >>> Omega, G_climate = calculate_climate_damage_and_gini_effect(
    ...     delta_T=2.0, Gini_current=0.4, y_mean=50000, params=params)
    >>> # Omega ≈ 0.08, G_climate ≈ 0.4 (approximately uniform)

    Income-dependent damage:
    >>> params['k_damage_halfsat'] = 10000
    >>> Omega, G_climate = calculate_climate_damage_and_gini_effect(
    ...     delta_T=2.0, Gini_current=0.4, y_mean=50000, params=params)
    >>> # Omega > 0.08, G_climate > 0.4 (inequality increases)
    """
    # Handle no-damage case
    if delta_T <= 0:
        return 0.0, Gini_current

    # Extract parameters
    k_coeff = params['k_damage_coeff']
    k_exp = params['k_damage_exp']
    k_halfsat = params['k_damage_halfsat']

    # Maximum damage fraction (uniform component)
    omega_max = k_coeff * (delta_T ** k_exp)

    # Special case: very high halfsat means approximately uniform damage
    if k_halfsat > 1e10:
        # Nearly uniform damage
        Omega_uniform = omega_max
        return Omega_uniform, Gini_current

    # Pareto parameter from Gini
    a = a_from_G(Gini_current)

    beta = (k_halfsat * (a - 1)) / (y_mean * a)
    Omega = omega_max * hyp2f1(1, a, a + 1, -beta)

    # Post-damage income distribution
    y_damaged = (1 - omega_F) * y_F

    # Calculate Gini of damaged distribution

    Gini_climate = calculate_climate_damage_gini_effect(
        a,                # a > 1  (Pareto-Lorenz and damage share 'a')
        k_halfsat,        # k >= 0
        y_mean,           # y > 0
        omega_max         # typically in [0,1]
        )
  
    return float(Omega), float(Gini_climate)

def calculate_climate_damage_gini_effect(
    a,                # a > 1  (Pareto-Lorenz and damage share 'a')
    k_halfsat,        # k >= 0
    y_mean,           # y > 0
    omega_max         # typically in [0,1]
    ):
    """
    Computes ΔG = G_new - G0 for Pareto–Lorenz L(F)=1-(1-F)^(1-1/a)
    with climate damage d(F) = ω * (1 - k/(k + y*a*(1-F)^(-1/a)/(a-1))).

    Closed forms (with α=a):
      S0   = (a-1)/(2a-1)
      G0   = 1/(2a-1)
      β    = k*(a-1)/(y*a)
      D    = ω * 2F1(1, a-1; a; -β)
      Sd   = ω * S0 * 2F1(1, 2a-1; 2a; -β)
      Gnew = 1 - 2*(S0 - Sd)/(1 - D)
      ΔG   = 2*(Sd - S0*D)/(1 - D)
    """
    if a <= 1:
        raise ValueError("a must be > 1 (ensures finite Pareto Gini and damage term).")
    if y_mean <= 0 or k_halfsat < 0:
        raise ValueError("Require y_mean > 0 and k_halfsat >= 0.")

    S0 = (a - 1.0) / (2.0 * a - 1.0)
    G0 = 1.0 / (2.0 * a - 1.0)
    beta = (k_halfsat * (a - 1.0)) / (y_mean * a)

    D  = omega_max * hyp2f1(1.0, a - 1.0, a, -beta)
    Sd = omega_max * S0 * hyp2f1(1.0, 2.0 * a - 1.0, 2.0 * a, -beta)

    G_new = 1.0 - 2.0 * (S0 - Sd) / (1.0 - D)
    return float(G_new)
