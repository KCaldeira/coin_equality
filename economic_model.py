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
    From equation (1.9): dK/dt = s·Y_net(t) - δ·K(t)
    """
    return s * Y_net - delta * K


def calculate_tendencies(state, params):
    """
    Calculate time derivatives of state variables.

    Parameters
    ----------
    state : dict
        State variables:
        - 'K': Capital stock ($)
        - 'Ecum': Cumulative CO2 emissions (tCO2)
    params : dict
        Model parameters (all must be provided):
        - 'alpha': Output elasticity of capital
        - 'delta': Capital depreciation rate (yr^-1)
        - 's': Savings rate
        - 'k_damage': Climate damage coefficient (°C^-β)
        - 'beta': Climate damage exponent
        - 'k_climate': Temperature sensitivity (°C tCO2^-1)
        - 'A': Total factor productivity (current)
        - 'L': Population (current)
        - 'sigma': Carbon intensity of GDP (current, tCO2 $^-1)
        - 'theta1': Abatement cost coefficient (current)
        - 'theta2': Abatement cost exponent
        - 'G1': Initial Gini index
        - 'delta_L': Fraction of income to redistribute
        - 'f': Fraction allocated to abatement vs redistribution

    Returns
    -------
    dict
        Time derivatives:
        - 'K': dK/dt ($ yr^-1)
        - 'Ecum': dEcum/dt = E (tCO2 yr^-1)

    Notes
    -----
    Calculation order follows equations 1.1-1.9 and 2.1:
    1. Y_gross from K, L, A, α (Eq 1.1)
    2. ΔT from Ecum, k_climate (Eq 2.2)
    3. Ω from ΔT, k_damage, β (Eq 1.2)
    4. Y_net from Y_gross, Ω (Eq 1.3)
    5. y from Y_net, L, s (Eq 1.7)
    6. Δc from y, ΔL (Eq 4.3)
    7. μ from f, Δc, θ₁, θ₂ (Eq 1.4)
    8. Λ from θ₁, μ, θ₂ (Eq 1.5)
    9. E from σ, μ, Y_gross (Eq 2.1)
    10. dK/dt from s, Y_net, δ, K (Eq 1.9)
    """
    # Extract state variables
    K = state['K']
    Ecum = state['Ecum']

    # Extract parameters
    alpha = params['alpha']
    delta = params['delta']
    s = params['s']
    k_damage = params['k_damage']
    beta = params['beta']
    k_climate = params['k_climate']
    A = params['A']
    L = params['L']
    sigma = params['sigma']
    theta1 = params['theta1']
    theta2 = params['theta2']
    delta_L = params['delta_L']
    f = params['f']

    # Step 1: Calculate gross production (Eq 1.1)
    Y_gross = calculate_gross_production(K, L, A, alpha)

    # Step 2: Calculate temperature change (Eq 2.2)
    delta_T = calculate_temperature_change(Ecum, k_climate)

    # Step 3: Calculate climate damage fraction (Eq 1.2)
    Omega = calculate_climate_damage_fraction(delta_T, k_damage, beta)

    # Step 4: Calculate net production after climate damage (Eq 1.3)
    Y_net = (1 - Omega) * Y_gross

    # Step 5: Calculate mean per-capita income (Eq 1.7)
    y = (1 - s) * Y_net / L

    # Step 6: Calculate per-capita amount redistributed (Eq 4.3)
    delta_c = y * delta_L

    # Step 7: Calculate abatement fraction (Eq 1.4)
    mu = (f * delta_c / theta1) ** (1 / theta2)

    # Step 8: Calculate emissions (Eq 2.1)
    E = calculate_emissions(sigma, mu, Y_gross)

    # Step 9: Calculate capital tendency (Eq 1.9)
    dK_dt = calculate_capital_tendency(s, Y_net, delta, K)

    # Step 10: Return tendencies
    return {
        'K': dK_dt,
        'Ecum': E
    }
