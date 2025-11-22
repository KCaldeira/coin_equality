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
from scipy.special import hyp2f1, gammaincc, gamma
from constants import INVERSE_EPSILON, EPSILON
import numpy as np


def pareto_integral_scipy(c_mean, a, c_scale):
    """
    Computes the income-dependent damage scaling factor for aggregate damage.

    This integral represents how aggregate damage scales with mean income
    for a Pareto-distributed population. As c_mean increases relative to
    c_scale, the integral decreases, representing better adaptation capacity
    of wealthier societies.

    Parameters
    ----------
    c_mean : float
        Mean income (y_mean in model).
    a : float
        Pareto parameter (>1), derived from Gini coefficient.
    c_scale : float
        Income scale for damage saturation (y_damage_aggregate_scale).

    Returns
    -------
    float
        Scaling factor for aggregate damage (0 to 1, decreasing with wealth).
    """
    # k = c_mean * (1 - 1/a)
    k = c_mean * (1.0 - 1.0 / a)

    # s = 1 - a
    s = 1.0 - a

    # Argument of the incomplete gamma
    x = k / c_scale

    # Upper incomplete gamma Γ(s, x)
    # SciPy: Γ(s, x) = gammaincc(s, x) * gamma(s)
    gamma_upper = gammaincc(s, x) * gamma(s)

    # Full analytic expression
    result = a * (k ** a) * (c_scale ** (1.0 - a)) * gamma_upper

    return result


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

