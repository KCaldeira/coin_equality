"""
Functions for calculating economic production, climate impacts, and system tendencies.

This module implements the Solow-Swann growth model with climate damage
and emissions abatement costs.
"""

import numpy as np
from income_distribution import G2_effective_pareto
from parameters import evaluate_params_at_time


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


def calculate_effective_gini(f, deltaL, G1):
    """
    Calculate effective Gini index after partial redistribution.

    Parameters
    ----------
    f : float
        Fraction allocated to abatement (vs redistribution)
    deltaL : float
        Fraction of income available for redistribution
    G1 : float
        Initial Gini index

    Returns
    -------
    float
        Effective Gini index after allocation

    Notes
    -----
    From equation (4.4): Uses Pareto-preserving two-step approach.
    When f=0 (all to redistribution), Gini is minimized.
    When f=1 (all to abatement), Gini is maximized.
    """
    G_eff, _ = G2_effective_pareto(f, deltaL, G1)
    return G_eff


def calculate_mean_utility(y_eff, G_eff, eta):
    """
    Calculate mean population utility using CRRA utility function.

    Parameters
    ----------
    y_eff : float
        Effective per-capita income (after abatement costs)
    G_eff : float
        Effective Gini index
    eta : float
        Coefficient of relative risk aversion

    Returns
    -------
    float
        Mean utility of the population

    Notes
    -----
    From equation (3.5):
    For η ≠ 1:
        U = [y^(1-η)/(1-η)] · [(1+G)^η(1-G)^(1-η)/(1+G(2η-1))]^(1/(1-η))

    For η = 1 (logarithmic utility):
        U = ln(y) + ln((1-G)/(1+G)) + 2G/(1+G)
    """
    if np.abs(eta - 1.0) < 1e-10:
        return np.log(y_eff) + np.log((1 - G_eff) / (1 + G_eff)) + 2 * G_eff / (1 + G_eff)

    term1 = (y_eff ** (1 - eta)) / (1 - eta)
    numerator = ((1 + G_eff) ** eta) * ((1 - G_eff) ** (1 - eta))
    denominator = 1 + G_eff * (2 * eta - 1)
    term2 = (numerator / denominator) ** (1 / (1 - eta))
    return term1 * term2


def calculate_tendencies(state, params):
    """
    Calculate time derivatives and all derived variables.

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
        - 'eta': Coefficient of relative risk aversion
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
        Dictionary containing:
        - Tendencies: 'dK_dt', 'dEcum_dt'
        - All intermediate variables: Y_gross, delta_T, Omega, Y_net, y, delta_c,
          mu, Lambda, abatecost, y_eff, G_eff, U, E

    Notes
    -----
    Calculation order follows equations 1.1-1.9, 2.1-2.2, 3.5, 4.3-4.4:
    1. Y_gross from K, L, A, α (Eq 1.1)
    2. ΔT from Ecum, k_climate (Eq 2.2)
    3. Ω from ΔT, k_damage, β (Eq 1.2)
    4. Y_net from Y_gross, Ω (Eq 1.3)
    5. y from Y_net, L, s (Eq 1.7)
    6. Δc from y, ΔL (Eq 4.3)
    7. μ from f, Δc, θ₁, θ₂ (Eq 1.4)
    8. Λ from θ₁, μ, θ₂ (Eq 1.5)
    9. abatecost from Λ, Y_net (Eq 1.6)
    10. y_eff from y, abatecost, L (Eq 1.8)
    11. G_eff from f, ΔL, G₁ (Eq 4.4)
    12. U from y_eff, G_eff, η (Eq 3.5)
    13. E from σ, μ, Y_gross (Eq 2.1)
    14. dK/dt from s, Y_net, δ, K (Eq 1.9)
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
    eta = params['eta']
    A = params['A']
    L = params['L']
    sigma = params['sigma']
    theta1 = params['theta1']
    theta2 = params['theta2']
    delta_L = params['delta_L']
    G1 = params['G1']
    f = params['f']

    # Step 1: Calculate gross production (Eq 1.1)
    Y_gross = calculate_gross_production(K, L, A, alpha)

    # Step 2: Calculate temperature change (Eq 2.2)
    delta_T = calculate_temperature_change(Ecum, k_climate)

    # Step 3: Calculate climate damage fraction (Eq 1.2)
    Omega = calculate_climate_damage_fraction(delta_T, k_damage, beta)

    # Step 4: Calculate production after climate damage (Eq 1.3)
    Y_damaged = calculate_damaged_production(Y_gross, Omega)

    # Step 5: Calculate mean per-capita income (Eq 1.7)
    y = (1 - s) * Y_damaged / L

    # Step 6: Calculate per-capita amount redistributed (Eq 4.3)
    delta_c = y * delta_L

    # Step 7: Calculate abatement fraction (Eq 1.4)
    mu = (f * delta_c / theta1) ** (1 / theta2)

    # Step 8: Calculate abatement cost fraction (Eq 1.5)
    Lambda = calculate_abatement_cost_fraction(mu, theta1, theta2)

    # Step 9: Calculate abatement cost (Eq 1.6)
    abatecost = Lambda * Y_damaged

    # Step 10: Calculate net production after abatement costs
    Y_net = calculate_net_production(Y_damaged, Lambda)

    # Step 11: Calculate effective per-capita income (Eq 1.8)
    y_eff = y - abatecost / L

    # Step 12: Calculate effective Gini index (Eq 4.4)
    G_eff = calculate_effective_gini(f, delta_L, G1)

    # Step 13: Calculate mean utility (Eq 3.5)
    U = calculate_mean_utility(y_eff, G_eff, eta)

    # Step 14: Calculate emissions (Eq 2.1)
    E = calculate_emissions(sigma, mu, Y_gross)

    # Step 15: Calculate capital tendency (Eq 1.9)
    dK_dt = calculate_capital_tendency(s, Y_net, delta, K)

    return {
        'dK_dt': dK_dt,
        'dEcum_dt': E,
        'Y_gross': Y_gross,
        'delta_T': delta_T,
        'Omega': Omega,
        'Y_damaged': Y_damaged,
        'Y_net': Y_net,
        'y': y,
        'delta_c': delta_c,
        'mu': mu,
        'Lambda': Lambda,
        'abatecost': abatecost,
        'y_eff': y_eff,
        'G_eff': G_eff,
        'U': U,
        'E': E,
    }


def integrate_model(config):
    """
    Integrate the model forward in time using Euler's method.

    Parameters
    ----------
    config : ModelConfiguration
        Complete model configuration including initial state, parameters,
        and time-dependent functions

    Returns
    -------
    dict
        Time series results with keys:
        - 't': array of time points
        - 'K': array of capital stock values
        - 'Ecum': array of cumulative emissions values
        - 'A', 'L', 'sigma', 'theta1', 'f': time-dependent inputs
        - All derived variables: Y_gross, delta_T, Omega, Y_damaged, Y_net,
          y, delta_c, mu, Lambda, abatecost, y_eff, G_eff, U, E
        - 'dK_dt', 'dEcum_dt': tendencies

    Notes
    -----
    Uses simple Euler integration: state(t+dt) = state(t) + dt * tendency(t)
    This ensures all functional relationships are satisfied exactly at output points.
    """
    # Extract integration parameters
    t_start = config.integration_params.t_start
    t_end = config.integration_params.t_end
    dt = config.integration_params.dt

    # Create time array
    t_array = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t_array)

    # Initialize state
    state = config.initial_state.copy()

    # Initialize storage for all variables
    results = {
        't': t_array,
        'K': np.zeros(n_steps),
        'Ecum': np.zeros(n_steps),
        'A': np.zeros(n_steps),
        'L': np.zeros(n_steps),
        'sigma': np.zeros(n_steps),
        'theta1': np.zeros(n_steps),
        'f': np.zeros(n_steps),
        'Y_gross': np.zeros(n_steps),
        'delta_T': np.zeros(n_steps),
        'Omega': np.zeros(n_steps),
        'Y_damaged': np.zeros(n_steps),
        'Y_net': np.zeros(n_steps),
        'y': np.zeros(n_steps),
        'delta_c': np.zeros(n_steps),
        'mu': np.zeros(n_steps),
        'Lambda': np.zeros(n_steps),
        'abatecost': np.zeros(n_steps),
        'y_eff': np.zeros(n_steps),
        'G_eff': np.zeros(n_steps),
        'U': np.zeros(n_steps),
        'E': np.zeros(n_steps),
        'dK_dt': np.zeros(n_steps),
        'dEcum_dt': np.zeros(n_steps),
    }

    # Time stepping loop
    for i, t in enumerate(t_array):
        # Evaluate time-dependent parameters at current time
        params = evaluate_params_at_time(t, config)

        # Calculate all variables and tendencies at current time
        outputs = calculate_tendencies(state, params)

        # Store state variables
        results['K'][i] = state['K']
        results['Ecum'][i] = state['Ecum']

        # Store time-dependent inputs
        results['A'][i] = params['A']
        results['L'][i] = params['L']
        results['sigma'][i] = params['sigma']
        results['theta1'][i] = params['theta1']
        results['f'][i] = params['f']

        # Store all derived variables
        results['Y_gross'][i] = outputs['Y_gross']
        results['delta_T'][i] = outputs['delta_T']
        results['Omega'][i] = outputs['Omega']
        results['Y_damaged'][i] = outputs['Y_damaged']
        results['Y_net'][i] = outputs['Y_net']
        results['y'][i] = outputs['y']
        results['delta_c'][i] = outputs['delta_c']
        results['mu'][i] = outputs['mu']
        results['Lambda'][i] = outputs['Lambda']
        results['abatecost'][i] = outputs['abatecost']
        results['y_eff'][i] = outputs['y_eff']
        results['G_eff'][i] = outputs['G_eff']
        results['U'][i] = outputs['U']
        results['E'][i] = outputs['E']
        results['dK_dt'][i] = outputs['dK_dt']
        results['dEcum_dt'][i] = outputs['dEcum_dt']

        # Euler step: update state for next iteration (skip on last step)
        if i < n_steps - 1:
            state['K'] = state['K'] + dt * outputs['dK_dt']
            state['Ecum'] = state['Ecum'] + dt * outputs['dEcum_dt']

    return results
