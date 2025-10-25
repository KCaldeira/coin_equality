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

The damage function uses a half-saturation (Michaelis-Menten) model:
    ω(y) = ω_max · k_halfsat / (k_halfsat + y)

where lower-income individuals (smaller y) experience higher fractional losses.

Analytical Solutions
--------------------
Rather than numerical integration, this module uses closed-form analytical
solutions obtained by integrating over the Pareto distribution. The key
mathematical tool is the Gauss hypergeometric function ₂F₁, which naturally
arises from the integral of the half-saturation damage function over the
Pareto income distribution.

These analytical solutions are:
- Exact (within numerical precision of hypergeometric function)
- Computationally efficient (no discretization needed)
- Numerically stable across wide parameter ranges

References: Analytical solutions derived with assistance from ChatGPT (2025).
"""

from income_distribution import a_from_G
from mpmath import hyp2f1


def calculate_climate_damage_and_gini_effect(delta_T, Gini_current, y_mean, params):
    """
    Calculate income-dependent climate damage and its effect on inequality.

    Uses analytical closed-form solutions based on hypergeometric functions.
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
        - 'psi1': linear damage coefficient (°C⁻¹)
        - 'psi2': quadratic damage coefficient (°C⁻²)
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
    **Damage Function (Half-Saturation Model):**
        ω_max(ΔT) = psi1 · ΔT + psi2 · ΔT²  [Barrage & Nordhaus 2023]
        ω(y, ΔT) = ω_max · k_halfsat / (k_halfsat + y)
                 = ω_max · (1 - y / (k_halfsat + y))

    where:
    - At income y = 0: ω = ω_max (maximum damage for poorest)
    - At income y = k_damage_halfsat: ω = ω_max/2 (half of maximum damage)
    - As income y → ∞: ω → 0 (damage approaches zero for wealthy)

    **Analytical Solution for Aggregate Damage:**
    For Pareto income distribution y(F) = ȳ · (1 - 1/a) · (1-F)^(-1/a), the
    aggregate damage is computed analytically as:

        β = k_halfsat · a / (ȳ · (a-1))
        Ω = ω_max · (k_halfsat / ȳ) · ₂F₁(1, a, a+1, -β)

    where ₂F₁ is the Gauss hypergeometric function. This closed-form solution
    replaces numerical integration and is exact (within numerical precision).

    **Analytical Solution for Post-Damage Gini:**
    The Gini index after climate damage is computed using hypergeometric functions
    that account for both the change in total income and the change in income
    distribution. See calculate_climate_damage_gini_effect() for details.

    **Special cases:**
    - k_damage_halfsat → ∞: β → 0, ₂F₁(1,a,a+1,0) = 1, uniform damage
    - k_damage_halfsat → 0: β → ∞, maximum regressive damage
    - ΔT = 0: ω_max = 0, no damage, Omega = 0, Gini_climate = Gini_current
    - a → ∞: Gini → 0 (perfect equality), damage becomes uniform

    **Computational advantages:**
    - No discretization or numerical integration required
    - Exact results (within machine precision)
    - Fast evaluation suitable for optimization loops
    - Numerically stable across wide parameter ranges

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

    # Handle degenerate Gini cases
    if Gini_current <= 0 or Gini_current >= 1:
        # Fall back to uniform damage if Gini is at boundary
        # Barrage & Nordhaus (2023) quadratic damage function
        Omega_uniform = params['psi1'] * delta_T + params['psi2'] * (delta_T ** 2)
        return Omega_uniform, Gini_current

    # Extract parameters
    psi1 = params['psi1']
    psi2 = params['psi2']
    k_halfsat = params['k_damage_halfsat']

    # Maximum damage fraction (temperature-dependent component)
    # Barrage & Nordhaus (2023) quadratic damage function:
    # ω_max(ΔT) = psi1 · ΔT + psi2 · ΔT²
    omega_max = psi1 * delta_T + psi2 * (delta_T ** 2)

    # Special case: very high halfsat means approximately uniform damage
    # When k_halfsat >> income, ω(y) ≈ ω_max for all y
    if k_halfsat > 1e10:
        # Nearly uniform damage: all income levels experience ~ω_max
        Omega_uniform = omega_max
        return Omega_uniform, Gini_current

    # Convert Gini to Pareto parameter
    # For Pareto distribution: Gini = 1/(2a-1), so a = (1+1/Gini)/2
    a = a_from_G(Gini_current)

    # ═══════════════════════════════════════════════════════════════════
    # ANALYTICAL SOLUTION FOR AGGREGATE DAMAGE
    # ═══════════════════════════════════════════════════════════════════
    #
    # The aggregate damage Ω is the income-weighted average of ω(y) over
    # the income distribution:
    #
    #   Ω = ∫ ω(y(F)) · y(F) · dF / ∫ y(F) · dF
    #
    # For Pareto distribution y(F) = ȳ·(1-1/a)·(1-F)^(-1/a) and
    # damage function ω(y) = ω_max · k/(k+y), this integral has the
    # closed-form solution:
    #
    #   Ω = ω_max · (k/ȳ) · ₂F₁(1, a, a+1, -β)
    #
    # where β = k·a/(ȳ·(a-1)) is a dimensionless parameter and
    # ₂F₁ is the Gauss hypergeometric function.
    #
    # Derivation: The integral reduces to a beta function integral that
    # can be expressed in terms of ₂F₁. This avoids numerical integration.
    # ═══════════════════════════════════════════════════════════════════

    # Dimensionless parameter β controlling damage distribution
    # β large → damage concentrated on poor; β small → more uniform
    beta = k_halfsat * a / (y_mean * (a - 1.0))

    # Aggregate damage via hypergeometric function
    # ₂F₁(1, a, a+1, -β) evaluated using mpmath for high precision
    Omega = omega_max * (k_halfsat / y_mean) * float(hyp2f1(1, a, a + 1, -beta))

    # ═══════════════════════════════════════════════════════════════════
    # ANALYTICAL SOLUTION FOR POST-DAMAGE GINI
    # ═══════════════════════════════════════════════════════════════════
    #
    # After applying income-dependent damage ω(y), the income distribution
    # becomes y_damaged(F) = (1 - ω(y(F))) · y(F).
    #
    # The Gini index of this damaged distribution is computed analytically
    # using hypergeometric functions. This accounts for:
    #   1. Total income reduction (by factor 1-Ω)
    #   2. Distributional shift (poor lose proportionally more)
    #
    # See calculate_climate_damage_gini_effect() for mathematical details.
    # ═══════════════════════════════════════════════════════════════════

    Gini_climate = calculate_climate_damage_gini_effect(
        a,           # Pareto parameter (a > 1)
        k_halfsat,   # Half-saturation income level ($)
        y_mean,      # Mean income before damage ($)
        omega_max    # Maximum damage fraction (0 ≤ ω_max < 1)
    )

    return float(Omega), float(Gini_climate)

def calculate_climate_damage_gini_effect(a, k_halfsat, y_mean, omega_max):
    """
    Compute post-damage Gini index using analytical solution.

    Calculates the Gini coefficient of the income distribution after applying
    income-dependent climate damage to a Pareto distribution. Uses closed-form
    expressions involving hypergeometric functions.

    Parameters
    ----------
    a : float
        Pareto parameter (a > 1). Related to pre-damage Gini by G₀ = 1/(2a-1).
    k_halfsat : float
        Income half-saturation for climate damage ($, k ≥ 0).
    y_mean : float
        Mean income before climate damage ($, y > 0).
    omega_max : float
        Maximum damage fraction (typically 0 ≤ ω ≤ 1).

    Returns
    -------
    G_new : float
        Gini index after climate damage is applied.

    Notes
    -----
    **Mathematical Framework:**

    For a Pareto distribution with Lorenz curve L(F) = 1 - (1-F)^(1-1/a),
    applying damage ω(y) = ω_max · k/(k+y) creates a new distribution:

        y_damaged(F) = [1 - ω(y(F))] · y(F)

    The Gini index of this damaged distribution is computed using:

    **Key Quantities:**
        S₀ = (a-1)/(2a-1)        - Integral of undamaged Lorenz curve
        G₀ = 1/(2a-1)            - Undamaged Gini coefficient
        β  = k·a/(ȳ·(a-1))       - Dimensionless damage parameter

    **Damage Integrals (via hypergeometric functions):**
        D  = ω · (k/ȳ) · ₂F₁(1, a, a+1, -β)                     - Aggregate damage fraction
        Sᵈ = ω · (a·k/ȳ) · (1/(2a-1)) · ₂F₁(1, 2a-1, 2a, -β)   - Damage-weighted Lorenz integral

    **Post-Damage Gini:**
        G_new = 1 - 2·(S₀ - Sᵈ)/(1 - D)

    The change in Gini is:
        ΔG = G_new - G₀ = 2·(Sᵈ - S₀·D)/(1 - D)

    This formula captures two effects:
    1. Total income reduction (denominator: 1-D)
    2. Regressive damage distribution (numerator: Sᵈ ≠ S₀·D when β > 0)

    **Physical Interpretation:**
    - When β → 0 (k_halfsat → ∞): uniform damage, Sᵈ = S₀·D, so ΔG = 0
    - When β > 0: poor suffer more, Sᵈ < S₀·D, so ΔG > 0 (inequality increases)
    - Larger β → more regressive damage → larger increase in Gini

    References
    ----------
    Analytical solution derived with assistance from ChatGPT (2025).
    Based on properties of hypergeometric functions and Lorenz curve integrals.
    """
    # Validate inputs
    if a <= 1:
        raise ValueError("a must be > 1 (ensures finite Pareto Gini and damage term).")
    if y_mean <= 0 or k_halfsat < 0:
        raise ValueError("Require y_mean > 0 and k_halfsat >= 0.")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Compute baseline Lorenz curve properties
    # ═══════════════════════════════════════════════════════════════════

    # S₀ = ∫₀¹ L(F) dF is the integral of the Lorenz curve
    # For Pareto with parameter a: S₀ = (a-1)/(2a-1)
    S0 = (a - 1.0) / (2.0 * a - 1.0)

    # G₀ = 1 - 2·S₀ is the Gini coefficient before damage
    # For Pareto: G₀ = 1/(2a-1)
    # (Not used in final calculation but shown for completeness)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Compute dimensionless damage parameter β
    # ═══════════════════════════════════════════════════════════════════

    # β = k·a/(ȳ·(a-1)) controls the regressivity of climate damage
    # - β → 0: damage becomes uniform (k_halfsat → ∞)
    # - β → ∞: damage highly concentrated on poor (k_halfsat → 0)
    beta = k_halfsat * a / (y_mean * (a - 1.0))

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: Compute damage integrals using hypergeometric functions
    # ═══════════════════════════════════════════════════════════════════

    # D = ∫₀¹ ω(y(F)) · y(F) dF / ∫₀¹ y(F) dF
    # This is the aggregate damage fraction (same as Omega from main function)
    # Analytical form: D = ω · (k/ȳ) · ₂F₁(1, a, a+1, -β)
    D = omega_max * (k_halfsat / y_mean) * float(hyp2f1(1.0, a, a + 1.0, -beta))

    # Sᵈ = ∫₀¹ L_damaged(F) dF where L_damaged is the Lorenz curve of
    # the damaged income distribution
    # Analytical form: Sᵈ = ω · (a·k/ȳ) · (1/(2a-1)) · ₂F₁(1, 2a-1, 2a, -β)
    Sd = omega_max * (a * k_halfsat / y_mean) * (1.0 / (2.0 * a - 1.0)) \
         * float(hyp2f1(1.0, 2.0 * a - 1.0, 2.0 * a, -beta))

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: Compute post-damage Gini coefficient
    # ═══════════════════════════════════════════════════════════════════

    # The Gini coefficient of the damaged distribution is:
    # G_new = 1 - 2·∫₀¹ L_damaged(F) dF
    #
    # The damaged Lorenz curve integral scales as:
    # ∫ L_damaged dF = (S₀ - Sᵈ)/(1 - D)
    #
    # Therefore:
    # G_new = 1 - 2·(S₀ - Sᵈ)/(1 - D)
    #
    # This can also be written as:
    # G_new = G₀ + ΔG where ΔG = 2·(Sᵈ - S₀·D)/(1 - D)
    #
    # When damage is uniform (β=0): Sᵈ = S₀·D, so ΔG = 0 (Gini unchanged)
    # When damage is regressive (β>0): Sᵈ < S₀·D, so ΔG > 0 (Gini increases)
    G_new = 1.0 - 2.0 * (S0 - Sd) / (1.0 - D)

    return float(G_new)
