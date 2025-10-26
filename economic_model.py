"""
Functions for calculating economic production, climate impacts, and system tendencies.

This module implements the Solow-Swann growth model with climate damage
and emissions abatement costs.
"""

import numpy as np
from income_distribution import calculate_Gini_effective_redistribute_abate
from parameters import evaluate_params_at_time
from climate_damage_distribution import calculate_climate_damage_and_gini_effect
from constants import EPSILON, NEG_BIGNUM


def calculate_tendencies(state, params):
    """
    Calculate time derivatives and all derived variables.

    Parameters
    ----------
    state : dict
        State variables:
        - 'K': Capital stock ($)
        - 'Ecum': Cumulative CO2 emissions (tCO2)
        - 'Gini': Current Gini index
    params : dict
        Model parameters (all must be provided):
        - 'alpha': Output elasticity of capital
        - 'delta': Capital depreciation rate (yr^-1)
        - 's': Savings rate
        - 'psi1': Linear climate damage coefficient (°C⁻¹) [Barrage & Nordhaus 2023]
        - 'psi2': Quadratic climate damage coefficient (°C⁻²) [Barrage & Nordhaus 2023]
        - 'y_damage_halfsat': Income half-saturation for climate damage ($)
        - 'k_climate': Temperature sensitivity (°C tCO2^-1)
        - 'eta': Coefficient of relative risk aversion
        - 'A': Total factor productivity (current)
        - 'L': Population (current)
        - 'sigma': Carbon intensity of GDP (current, tCO2 $^-1)
        - 'theta1': Abatement cost coefficient (current, $ tCO2^-1)
        - 'theta2': Abatement cost exponent
        - 'Gini_initial': Initial Gini index
        - 'Gini_fract': Fraction of Gini change as instantaneous step
        - 'Gini_restore': Rate of restoration to Gini_initial (yr^-1)
        - 'delta_L': Fraction of income to redistribute
        - 'f': Fraction allocated to abatement vs redistribution

    Returns
    -------
    dict
        Dictionary containing:
        - Tendencies: 'dK_dt', 'dEcum_dt', 'dGini_dt', 'Gini_step_change'
        - All intermediate variables: Y_gross, delta_T, Omega, Y_net, y, delta_c,
          mu, Lambda, abatecost, y_eff, G_eff, U, E

    Notes
    -----
    Calculation order follows equations 1.1-1.10, 2.1-2.2, 3.5, 4.3-4.4:
    1. Y_gross from K, L, A, α (Eq 1.1)
    2. ΔT from Ecum, k_climate (Eq 2.2)
    3. y_gross from Y_gross, L (mean per-capita gross income)
    4. Ω, G_climate from ΔT, Gini, y_gross, damage params (income-dependent damage)
    5. Y_damaged from Y_gross, Ω (Eq 1.3)
    6. y from Y_damaged, L, s (Eq 1.4)
    7. Δc from y, ΔL (Eq 4.3)
    8. E_pot from σ, Y_gross (Eq 2.1)
    9. abatecost from f, Δc, L (Eq 1.5)
    10. μ from abatecost, θ₁, θ₂, E_pot (Eq 1.6)
    11. Λ from abatecost, Y_damaged (Eq 1.7)
    12. Y_net from Y_damaged, Λ (Eq 1.8)
    13. y_eff from y, abatecost, L (Eq 1.9)
    14. G_eff from f, ΔL, G_climate (Eq 4.4, applied to climate-damaged distribution)
    15. U from y_eff, G_eff, η (Eq 3.5)
    16. E from σ, μ, Y_gross (Eq 2.3)
    17. dK/dt from s, Y_net, δ, K (Eq 1.10)
    18. dGini/dt, Gini_step from Gini dynamics
    """
    # Extract state variables
    K = state['K']
    Ecum = state['Ecum']
    Gini = state['Gini']

    # Extract parameters
    alpha = params['alpha']
    delta = params['delta']
    s = params['s']
    k_climate = params['k_climate']
    eta = params['eta']
    A = params['A']
    L = params['L']
    sigma = params['sigma']
    theta1 = params['theta1']
    theta2 = params['theta2']
    delta_L = params['delta_L']
    Gini_initial = params['Gini_initial']
    Gini_fract = params['Gini_fract']
    Gini_restore = params['Gini_restore']
    f = params['f']

    # strange things can happen during the optimization phase, thus the if-then checks below

    # Eq 1.1: Gross production (Cobb-Douglas)
    if K>0:
        Y_gross = A * (K ** alpha) * (L ** (1 - alpha))
    else:
        Y_gross = 0.0

    # Eq 2.2: Temperature change from cumulative emissions
    delta_T = k_climate * Ecum

    # Mean per-capita gross income (before climate damage)
    if L > 0:
        y_gross = Y_gross / L
    else:
        y_gross = 0.0

    # Income-dependent climate damage
    if y_gross > 0:
    # Returns both aggregate damage fraction and post-damage Gini
        Omega, Gini_climate = calculate_climate_damage_and_gini_effect(
          delta_T, Gini, y_gross, params
      )
    else:
        Omega = 0.0
        Gini_climate = Gini

    # Clamp Gini_climate to valid bounds before using in subsequent calculations
    Gini_climate = np.clip(Gini_climate, EPSILON, 1.0 - EPSILON)

    # Eq 1.3: Production after climate damage
    Y_damaged = (1 - Omega) * Y_gross

    # Eq 1.4: Mean per-capita income (after climate damage, before abatement)
    y = (1 - s) * Y_damaged / L

    # Eq 4.3: Per-capita amount redistributed
    delta_c = y * delta_L

    # Eq 2.1: Potential emissions (unabated)
    Epot = sigma * Y_gross

    # Eq 1.5: Abatement cost (what society allocates to abatement)
    abatecost = f * delta_c * L

    # Eq 1.6: Abatement fraction
    if Epot > 0 and abatecost > 0:
        mu = (abatecost * theta2 / (Epot * theta1)) ** (1 / theta2)
    else:
        mu = 0.0

    # Eq 1.7: Abatement cost fraction
    if Y_damaged > 0 and abatecost > 0:
        Lambda = abatecost / Y_damaged
    else:
        Lambda = 0.0

    # Eq 1.8: Net production after abatement costs
    Y_net = (1 - Lambda) * Y_damaged

    # Eq 1.9: Effective per-capita income
    y_eff = y - abatecost / L

    # Eq 4.4: Effective Gini index
    # Redistribution operates on the climate-damaged distribution
    G_eff, _ = calculate_Gini_effective_redistribute_abate(f, delta_L, Gini_climate)

    # Eq 3.5: Mean utility
    if y_eff > 0 and 0 <= G_eff <= 1.0:
        if np.abs(eta - 1.0) < EPSILON:
            U = np.log(y_eff) + np.log((1 - G_eff) / (1 + G_eff)) + 2 * G_eff / (1 + G_eff)
        else:
            term1 = (y_eff ** (1 - eta)) / (1 - eta)
            numerator = ((1 + G_eff) ** eta) * ((1 - G_eff) ** (1 - eta))
            denominator = 1 + G_eff * (2 * eta - 1)
            term2 = (numerator / denominator) ** (1 / (1 - eta))
            U = term1 * term2
    else:
        U = NEG_BIGNUM

    # Eq 2.3: Actual emissions (after abatement)
    E = sigma * (1 - mu) * Y_gross

    # Eq 1.10: Capital tendency
    dK_dt = s * Y_net - delta * K

    # Gini dynamics
    dGini_dt = -Gini_restore * (Gini - Gini_initial)
    Gini_step_change = Gini_fract * (G_eff - Gini)

    return {
        'dK_dt': dK_dt,
        'dEcum_dt': E,
        'dGini_dt': dGini_dt,
        'Gini_step_change': Gini_step_change,
        'Y_gross': Y_gross,
        'delta_T': delta_T,
        'Omega': Omega,
        'Gini_climate': Gini_climate,
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
        - 'Gini': array of Gini index values
        - 'A', 'L', 'sigma', 'theta1', 'f': time-dependent inputs
        - All derived variables: Y_gross, delta_T, Omega, Gini_climate, Y_damaged, Y_net,
          y, delta_c, mu, Lambda, abatecost, y_eff, G_eff, U, E
        - 'dK_dt', 'dEcum_dt', 'dGini_dt', 'Gini_step_change': tendencies

    Notes
    -----
    Uses simple Euler integration: state(t+dt) = state(t) + dt * tendency(t)
    This ensures all functional relationships are satisfied exactly at output points.

    Initial conditions are computed automatically:
    - Ecum(0) = 0 (no cumulative emissions)
    - K(0) = (s·A(0)/δ)^(1/(1-α))·L(0) (steady-state capital)
    - Gini(0) = Gini_initial (initial Gini index from configuration)
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
        'Ecum': 0.0,
        'Gini': config.scalar_params.Gini_initial
    }

    # Initialize storage for all variables
    results = {
        't': t_array,
        'K': np.zeros(n_steps),
        'Ecum': np.zeros(n_steps),
        'Gini': np.zeros(n_steps),
        'A': np.zeros(n_steps),
        'L': np.zeros(n_steps),
        'sigma': np.zeros(n_steps),
        'theta1': np.zeros(n_steps),
        'f': np.zeros(n_steps),
        'Y_gross': np.zeros(n_steps),
        'delta_T': np.zeros(n_steps),
        'Omega': np.zeros(n_steps),
        'Gini_climate': np.zeros(n_steps),
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
        'dGini_dt': np.zeros(n_steps),
        'Gini_step_change': np.zeros(n_steps),
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
        results['Gini'][i] = state['Gini']

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
        results['Gini_climate'][i] = outputs['Gini_climate']
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
        results['dGini_dt'][i] = outputs['dGini_dt']
        results['Gini_step_change'][i] = outputs['Gini_step_change']

        # Euler step: update state for next iteration (skip on last step)
        if i < n_steps - 1:
            state['K'] = state['K'] + dt * outputs['dK_dt']
            # do not allow cumulative emissions to go negative, making it colder than the initial condition
            state['Ecum'] = max(0.0, state['Ecum'] + dt * outputs['dEcum_dt'])
            # Gini update includes both continuous change and discontinuous step
            # Clamp Gini to stay within valid bounds (0, 1) exclusive
            state['Gini'] = np.clip(
                state['Gini'] + dt * outputs['dGini_dt'] + outputs['Gini_step_change'],
                EPSILON,
                1.0 - EPSILON
            )

    return results
