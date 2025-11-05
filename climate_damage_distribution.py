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

Damage Function (Corrected)
---------------------------
The damage function follows a half-saturation (Michaelis–Menten) form:
    ω(y) = ω_max * (1 - y / (y_half + y))
         = ω_max * y_half / (y_half + y)

This means lower-income individuals (small y) experience proportionally higher
fractional losses, while high-income individuals experience smaller losses.

Analytical Solutions
--------------------
Closed-form analytical solutions use the Gauss hypergeometric function ₂F₁.

References: Analytical solutions derived with assistance from ChatGPT (2025).
"""

from income_distribution import a_from_G
from mpmath import hyp2f1
from constants import INVERSE_EPSILON, EPSILON


def calculate_climate_damage_and_gini_effect(delta_T, Gini_current, y_mean, params):
    """
    Calculate income-dependent climate damage and its effect on inequality.

    Uses analytical closed-form solutions based on hypergeometric functions.
    Climate damage is applied as a function of income level using a half-saturation
    (Michaelis–Menten) model:
        ω(y) = ω_max * (1 - y / (y_half + y)) = ω_max * y_half / (y_half + y)

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
        - 'y_damage_halfsat': income half-saturation constant ($)
        - 'fract_gdp': optional, fraction of GDP redistributed (default 0)

    Returns
    -------
    Omega : float
        Aggregate damage fraction (0 ≤ Ω < 1)
    Gini_climate : float
        Post-damage Gini index
    """
    if delta_T <= 0:
        return 0.0, Gini_current

    if Gini_current <= 0 or Gini_current >= 1:
        Omega_uniform = params['psi1'] * delta_T + params['psi2'] * (delta_T ** 2)
        return Omega_uniform, Gini_current

    psi1 = params['psi1']
    psi2 = params['psi2']
    y_half = params['y_damage_halfsat']
    fract_gdp = params.get('fract_gdp', 0)

    # Quadratic damage response (Barrage & Nordhaus, 2023)
    omega_max = psi1 * delta_T + psi2 * (delta_T ** 2)
    omega_max = min(omega_max, 1.0 - EPSILON)

    # Uniform damage special cases
    if y_half > INVERSE_EPSILON or fract_gdp >= 1:
        return omega_max, Gini_current

    # Convert Gini → Pareto parameter
    a = a_from_G(Gini_current)

    # Dimensionless parameter for regressivity
    b = (a * y_half) / ((a - 1.0) * y_mean)

    # === Aggregate damage (Ω) ===
    # Closed form: Ω = ω_max * (y_half / y_mean) * ₂F₁(1, a, a+1, -b)
    Omega = omega_max * (y_half / y_mean) * float(hyp2f1(1, a, a + 1, -b))

    # === Post-damage Gini (G_climate) ===
    Gini_climate = calculate_effect_of_climate_damage_on_gini_index(
        Gini_current, y_half, y_mean, omega_max
    )

    return float(Omega), float(Gini_climate)


def calculate_effect_of_climate_damage_on_gini_index(Gini_initial, y_half, y_mean, omega_max):
    """
    Compute post-damage Gini index analytically.

    Damage function:
        ω(y) = ω_max * (1 - y / (y_half + y))
             = ω_max * y_half / (y_half + y)

    Parameters
    ----------
    Gini_initial : float
        Pre-damage Gini index (0 < G < 1)
    y_half : float
        Half-saturation income ($)
    y_mean : float
        Mean income before damage ($)
    omega_max : float
        Maximum damage fraction (0 ≤ ω ≤ 1)

    Returns
    -------
    G_new : float
        Post-damage Gini index.
    """
    if Gini_initial <= 0 or Gini_initial >= 1:
        raise ValueError("Gini_initial must be in range (0, 1).")
    if y_mean <= 0 or y_half < 0:
        raise ValueError("Require y_mean > 0 and y_half ≥ 0.")

    # Convert Gini to Pareto parameter
    a = a_from_G(Gini_initial)

    # Baseline Gini
    G0 = Gini_initial

    # Dimensionless parameter controlling regressivity
    b = (a * y_half) / ((a - 1.0) * y_mean)

    # Hypergeometric terms
    Phi = float(hyp2f1(1.0, a - 1.0, a, -b))             # Mean damage factor
    H   = float(hyp2f1(1.0, 2.0 * a - 1.0, 2.0 * a, -b)) # Inequality adjustment

    omega_mean = omega_max * Phi
    denom = 1.0 - omega_mean

    if denom <= 0:
        # Degenerate case (complete collapse)
        return float(G0)

    # Closed-form post-damage Gini
    G_new = 1.0 - (1.0 - G0) * (1.0 - omega_max * H) / denom
    return float(G_new)
