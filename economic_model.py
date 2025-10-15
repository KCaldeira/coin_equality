"""
Functions for calculating economic production, climate impacts, and system tendencies.

This module implements the Solow-Swann growth model with climate damage
and emissions abatement costs.
"""

from income_distribution import G2_effective_pareto


def calculate_gross_production(K, L, A, alpha):
    """
    Calculate gross production using Cobb-Douglas production function.

    Parameters
    ----------
    K : float
        Capital stock ($)
    L : float
        Labor force / population (people)
    A : float
        Total factor productivity ($^(1-α) yr^-1 people^(α-1))
    alpha : float
        Output elasticity of capital

    Returns
    -------
    float
        Gross production ($ yr^-1)

    Notes
    -----
    From equation (18): Y_gross(t) = A(t) · K(t)^α · L(t)^(1-α)
    """
    return A * (K ** alpha) * (L ** (1 - alpha))


def calculate_temperature_change(Ecum, k_climate):
    """
    Calculate global mean temperature increase from cumulative emissions.

    Parameters
    ----------
    Ecum : float
        Cumulative CO2 emissions (tCO2)
    k_climate : float
        Temperature sensitivity to cumulative emissions (°C tCO2^-1)

    Returns
    -------
    float
        Global mean temperature increase (°C)

    Notes
    -----
    From equation (25): ΔT(t) = k_climate · ∫₀^t E(t') dt'
    Since we track Ecum = ∫ E dt, this simplifies to: ΔT = k_climate · Ecum
    """
    return k_climate * Ecum


def calculate_climate_damage_fraction(delta_T, k_damage, beta):
    """
    Calculate fraction of production lost to climate damage.

    Parameters
    ----------
    delta_T : float
        Global mean temperature increase (°C)
    k_damage : float
        Climate damage coefficient (°C^-β)
    beta : float
        Climate damage exponent

    Returns
    -------
    float
        Fraction of gross production lost to climate damage

    Notes
    -----
    From equation (21): Ω(t) = k_damage · ΔT(t)^β
    """
    return k_damage * (delta_T ** beta)


def calculate_damaged_production(Y_gross, Omega):
    """
    Calculate production after accounting for climate damage.

    Parameters
    ----------
    Y_gross : float
        Gross production ($ yr^-1)
    Omega : float
        Fraction of production lost to climate damage

    Returns
    -------
    float
        Production net of climate damage ($ yr^-1)

    Notes
    -----
    From equation (19): Y_damaged(t) = (1 - Ω(t)) · Y_gross(t)
    """
    return (1 - Omega) * Y_gross


def calculate_abatement_cost_fraction(mu, theta1, theta2):
    """
    Calculate fraction of gross production allocated to emissions abatement.

    Parameters
    ----------
    mu : float
        Fraction of emissions abated
    theta1 : float
        Abatement cost coefficient
    theta2 : float
        Abatement cost exponent

    Returns
    -------
    float
        Fraction of gross production allocated to abatement

    Notes
    -----
    From equation (24): Λ(t) = θ₁(t) · μ(t)^θ₂
    """
    return theta1 * (mu ** theta2)


def calculate_net_production(Y_damaged, Lambda):
    """
    Calculate production net of climate damage and abatement costs.

    Parameters
    ----------
    Y_damaged : float
        Production net of climate damage ($ yr^-1)
    Lambda : float
        Fraction of gross production allocated to abatement

    Returns
    -------
    float
        Net production ($ yr^-1)

    Notes
    -----
    From equation (20): Y_net(t) = (1 - Λ(t)) · Y_damaged(t)
    """
    return (1 - Lambda) * Y_damaged


def calculate_emissions(sigma, mu, Y_gross):
    """
    Calculate CO2 emissions net of abatement.

    Parameters
    ----------
    sigma : float
        Carbon intensity of GDP (tCO2 $^-1)
    mu : float
        Fraction of emissions abated
    Y_gross : float
        Gross production ($ yr^-1)

    Returns
    -------
    float
        CO2 emissions (tCO2 yr^-1)

    Notes
    -----
    From equation (23): E(t) = σ(t) · (1 - μ(t)) · Y_gross(t)
    """
    return sigma * (1 - mu) * Y_gross


def calculate_capital_tendency(s, Y_net, delta, K):
    """
    Calculate rate of change of capital stock.

    Parameters
    ----------
    s : float
        Savings rate
    Y_net : float
        Net production ($ yr^-1)
    delta : float
        Capital depreciation rate (yr^-1)
    K : float
        Capital stock ($)

    Returns
    -------
    float
        Rate of change of capital stock ($ yr^-1)

    Notes
    -----
    From equation (26): dK/dt = s·Y_net(t) - δ·K(t)
    """
    return s * Y_net - delta * K
