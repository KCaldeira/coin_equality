"""
Functions for calculating economic production, climate impacts, and system tendencies.

This module implements the Solow-Swann growth model with climate damage
and emissions abatement costs.
"""

import numpy as np
from income_distribution import G2_effective_pareto
from parameters import evaluate_params_at_time


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
        - 'k_damage_coeff': Climate damage coefficient (°C^-k_damage_exp)
        - 'k_damage_exp': Climate damage exponent
        - 'k_climate': Temperature sensitivity (°C tCO2^-1)
        - 'eta': Coefficient of relative risk aversion
        - 'A': Total factor productivity (current)
        - 'L': Population (current)
        - 'sigma': Carbon intensity of GDP (current, tCO2 $^-1)
        - 'theta1': Abatement cost coefficient (current, $ tCO2^-1)
        - 'theta2': Abatement cost exponent
        - 'Gini_initial': Initial Gini index
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
    Calculation order follows equations 1.1-1.10, 2.1-2.2, 3.5, 4.3-4.4:
    1. Y_gross from K, L, A, α (Eq 1.1)
    2. ΔT from Ecum, k_climate (Eq 2.2)
    3. Ω from ΔT, k_damage_coeff, k_damage_exp (Eq 1.2)
    4. Y_damaged from Y_gross, Ω (Eq 1.3)
    5. y from Y_damaged, L, s (Eq 1.4)
    6. Δc from y, ΔL (Eq 4.3)
    7. E_pot from σ, Y_gross (Eq 2.1)
    8. abatecost from f, Δc, L (Eq 1.5)
    9. μ from abatecost, θ₁, θ₂, E_pot (Eq 1.6)
    10. Λ from abatecost, Y_damaged (Eq 1.7)
    11. Y_net from Y_damaged, Λ (Eq 1.8)
    12. y_eff from y, abatecost, L (Eq 1.9)
    13. G_eff from f, ΔL, Gini_initial (Eq 4.4)
    14. U from y_eff, G_eff, η (Eq 3.5)
    15. E from σ, μ, Y_gross (Eq 2.3)
    16. dK/dt from s, Y_net, δ, K (Eq 1.10)
    """
    # Extract state variables
    K = state['K']
    Ecum = state['Ecum']

    # Extract parameters
    alpha = params['alpha']
    delta = params['delta']
    s = params['s']
    k_damage_coeff = params['k_damage_coeff']
    k_damage_exp = params['k_damage_exp']
    k_climate = params['k_climate']
    eta = params['eta']
    A = params['A']
    L = params['L']
    sigma = params['sigma']
    theta1 = params['theta1']
    theta2 = params['theta2']
    delta_L = params['delta_L']
    Gini_initial = params['Gini_initial']
    f = params['f']

    # Eq 1.1: Gross production (Cobb-Douglas)
    Y_gross = A * (K ** alpha) * (L ** (1 - alpha))

    # Eq 2.2: Temperature change from cumulative emissions
    delta_T = k_climate * Ecum

    # Eq 1.2: Climate damage fraction
    Omega = k_damage_coeff * (delta_T ** k_damage_exp)

    # Eq 1.3: Production after climate damage
    Y_damaged = (1 - Omega) * Y_gross

    # Eq 1.4: Mean per-capita income
    y = (1 - s) * Y_damaged / L

    # Eq 4.3: Per-capita amount redistributed
    delta_c = y * delta_L

    # Eq 2.1: Potential emissions (unabated)
    Epot = sigma * Y_gross

    # Eq 1.5: Abatement cost (what society allocates to abatement)
    abatecost = f * delta_c * L

    # Eq 1.6: Abatement fraction
    mu = (abatecost * theta2 / (Epot * theta1)) ** (1 / theta2)

    # Eq 1.7: Abatement cost fraction
    Lambda = abatecost / Y_damaged

    # Eq 1.8: Net production after abatement costs
    Y_net = (1 - Lambda) * Y_damaged

    # Eq 1.9: Effective per-capita income
    y_eff = y - abatecost / L

    # Eq 4.4: Effective Gini index
    G_eff, _ = G2_effective_pareto(f, delta_L, Gini_initial)

    # Eq 3.5: Mean utility
    U_failure = -1e20
    if y_eff <= 0:
        U = U_failure
    elif np.abs(eta - 1.0) < 1e-10:
        U = np.log(y_eff) + np.log((1 - G_eff) / (1 + G_eff)) + 2 * G_eff / (1 + G_eff)
    else:
        term1 = (y_eff ** (1 - eta)) / (1 - eta)
        numerator = ((1 + G_eff) ** eta) * ((1 - G_eff) ** (1 - eta))
        denominator = 1 + G_eff * (2 * eta - 1)
        term2 = (numerator / denominator) ** (1 / (1 - eta))
        U = term1 * term2

    # Eq 2.3: Actual emissions (after abatement)
    E = sigma * (1 - mu) * Y_gross

    # Eq 1.10: Capital tendency
    dK_dt = s * Y_net - delta * K

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
        Complete model configuration including parameters and time-dependent functions

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

    Initial conditions are computed automatically:
    - Ecum(0) = 0 (no cumulative emissions)
    - K(0) = (s·A(0)/δ)^(1/(1-α))·L(0) (steady-state capital)
    """
    # Extract integration parameters
    t_start = config.integration_params.t_start
    t_end = config.integration_params.t_end
    dt = config.integration_params.dt

    # Create time array
    t_array = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t_array)

    # Calculate initial state
    A0 = config.time_functions['A'](t_start)
    L0 = config.time_functions['L'](t_start)
    s = config.scalar_params.s
    delta = config.scalar_params.delta
    alpha = config.scalar_params.alpha

    K0 = ((s * A0 / delta) ** (1 / (1 - alpha))) * L0

    state = {
        'K': K0,
        'Ecum': 0.0
    }

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
