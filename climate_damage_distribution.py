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
from scipy.special import hyp2f1
from constants import INVERSE_EPSILON, EPSILON
import numpy as np


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

    # Quadratic damage response (Barrage & Nordhaus, 2023)
    omega_max = psi1 * delta_T + psi2 * (delta_T ** 2)
    omega_max = min(omega_max, 1.0 - EPSILON)

    # Uniform damage special case: large y_half means damage is uniform across income levels
    if y_half > INVERSE_EPSILON:
        return omega_max, Gini_current

    # Convert Gini → Pareto parameter
    a = a_from_G(Gini_current)
    lorenz_exponent = 1.0 - 1.0 / a

    # Dimensionless parameter for regressivity
    b =  y_half / (y_mean * lorenz_exponent)

    # === Aggregate damage (Ω) ===
    # Closed form: Ω = ω_max * (y_half / y_mean) * ₂F₁(1, a, a+1, -b)
    # Using scipy.special.hyp2f1 (218x faster than mpmath, same accuracy)
    H1 = hyp2f1(1.0, a, a + 1.0, -b)  # Mean damage factor
    H2 = hyp2f1(1.0, 2.0 * a, 2.0 * a + 1.0, -b)  # Inequality adjustment

    omega_max_scaled = omega_max * (y_half / y_mean)
    Omega = omega_max_scaled * H1

    # === Post-damage Gini (G_climate) ===
    Gini_climate = (Gini_current + omega_max_scaled * (H2 - H1)) / (1.0 - omega_max_scaled * H1)
    if np.isnan(Gini_climate) or Gini_climate < 0.0:
        print("Warning: Gini_climate computation produced invalid value. Setting to 0.0.")
        print(f"  Inputs: delta_T={delta_T}, Gini_current={Gini_current}, y_mean={y_mean}, params={params}")
        print(f"  Computed: Omega={Omega}, Gini_climate (raw)={Gini_climate}")
        Gini_climate = 0.0

    return float(Omega), float(Gini_climate)

