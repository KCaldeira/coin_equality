"""
Functions for calculating economic production, climate impacts, and system tendencies.

This module implements the Solow-Swann growth model with climate damage
and emissions abatement costs.
"""

import numpy as np
from income_distribution import calculate_Gini_effective_redistribute_abate
from parameters import evaluate_params_at_time
from climate_damage_distribution import calculate_climate_damage_and_gini_effect
from constants import EPSILON, LOOSE_EPSILON, NEG_BIGNUM, MAX_INITIAL_CAPITAL_ITERATIONS


def calculate_tendencies(state, params, store_detailed_output=True):
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
        - 'y_damage_distribution_scale': Income half-saturation for climate damage ($)
        - 'k_climate': Temperature sensitivity (°C tCO2^-1)
        - 'eta': Coefficient of relative risk aversion
        - 'A': Total factor productivity (current)
        - 'L': Population (current)
        - 'sigma': Carbon intensity of GDP (current, tCO2 $^-1)
        - 'theta1': Abatement cost coefficient (current, $ tCO2^-1)
        - 'theta2': Abatement cost exponent
        - 'mu_max': Maximum allowed abatement fraction (cap on μ)
        - 'Gini_background': Background Gini index (current, from time function)
        - 'Gini_fract': Fraction of Gini change as instantaneous step
        - 'Gini_restore': Rate of restoration to Gini_background (yr^-1)
        - 'fract_gdp': Fraction of GDP available for redistribution and abatement
        - 'f': Fraction allocated to abatement vs redistribution
    store_detailed_output : bool, optional
        Whether to compute and return all intermediate variables. Default: True

    Returns
    -------
    dict
        Dictionary containing:
        - Tendencies: 'dK_dt', 'dEcum_dt', 'd_delta_Gini_dt', 'delta_Gini_step_change'
        - All intermediate variables: Y_gross, delta_T, Omega, Y_net, y, redistribution,
          mu, Lambda, AbateCost, y_net, G_eff, U, E

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
    9. AbateCost from f, Δc, L (Eq 1.5)
    10. μ from AbateCost, θ₁, θ₂, E_pot (Eq 1.6)
    11. Λ from AbateCost, Y_damaged (Eq 1.7)
    12. Y_net from Y_damaged, Λ (Eq 1.8)
    13. y_net from y, AbateCost, L (Eq 1.9)
    14. G_eff from f, ΔL, G_climate (Eq 4.4, applied to climate-damaged distribution)
    15. U from y_net, G_eff, η (Eq 3.5)
    16. E from σ, μ, Y_gross (Eq 2.3)
    17. dK/dt from s, Y_net, δ, K (Eq 1.10)
    18. d(delta_Gini)/dt, Gini_step from Gini dynamics
    """
    # Extract state variables
    K = state['K']
    Ecum = state['Ecum']
    delta_Gini = state['delta_Gini']

    # Extract parameters
    alpha = params['alpha']
    delta = params['delta']
    s = params['s']
    k_climate = params['k_climate']
    eta = params['eta']
    rho = params['rho']
    t = params['t']
    A = params['A']
    L = params['L']
    sigma = params['sigma']
    theta1 = params['theta1']
    theta2 = params['theta2']
    mu_max = params['mu_max']
    fract_gdp = params['fract_gdp']
    Gini_background = params['Gini_background']
    Gini_fract = params['Gini_fract']
    Gini_restore = params['Gini_restore']
    f = params['f']

    # Policy switches
    income_dependent_aggregate_damage = params['income_dependent_aggregate_damage']
    income_dependent_damage_distribution = params['income_dependent_damage_distribution']
    tax_policy_type = params['tax_policy_type']
    redistribution_policy_type = params['redistribution_policy_type']

    #========================================================================================
    # First calculate things that don't depend on climate damage or redistribution or taxation

    # strange things can happen during the optimization phase, thus the if-then checks below

    # Eq 1.1: Gross production (Cobb-Douglas)
    if K>0:
        Y_gross = A * (K ** alpha) * (L ** (1 - alpha))
    else:
        Y_gross = 0.0

    # Mean per-capita gross income (before climate damage)
    if L > 0:
        y_gross = Y_gross / L
    else:
        y_gross = 0.0


    # Eq 2.2: Temperature change from cumulative emissions
    delta_T = k_climate * Ecum

    # Base climate damage from temperature (before income-dependent adjustments)
    omega_base = params['psi1'] * delta_T + params['psi2'] * delta_T**2


    #========================================================================================
    # Now calculate things that do depend on climate damage or redistribution or taxation

    n_iterations = 0
    converged = False
    y_net_prev_prev = None
    y_net_new_prev = None

    while n_iterations < MAX_INITIAL_CAPITAL_ITERATIONS and not converged:
        y_net_prev = y_net
        n_iterations += 1


    #------------------------------------------------------------------------------------------
    # Taxation
    # taxation is assumed to be a fraction of gross income after climate damage but before abatement and redistribution
    if tax_policy_type == 'uniform_fractional':
        # Uniform carbon tax (no redistribution effect)
        def tax(F,y, fract_gdp, Gini, mean_damage, damage_dist_fn):
            Gini_index_part = fract_gdp * y * ((1 - Gini)/(1 + Gini))*(1 - F)**(-2 Gini/(1 + Gini))
            damage_part = fract_gdp * mean_damage * damage_dist_fn(F)
            return Gini_index_part - damage_part
    elif tax_policy_type == 'tax_richest':
        # Tax only the richest fraction of the population
        # this would be the critical value above which people would be taxed without climate damage
        fcrit_first_guess = 1.0-(fract_gdp*(1 + Gini)/(2* Gini))**((1 + Gini)/(1-Gini))
    else:
        #raise error
        raise ValueError(f"Unknown tax_policy_type '{tax_policy_type}'.")

    #------------------------------------------------------------------------------------------
    # Redistribution

    #------------------------------------------------------------------------------------------
    # Climate damage

    #------------------------------------------------------------------------------------------
    # Abatement

    #------------------------------------------------------------------------------------------
    # Remaining calculations
    Climate_Damage = Omega * Y_gross
    Y_damaged = Y_gross - Climate_Damage

    Lambda = f * fract_gdp
    AbateCost = Lambda * Y_damaged
    Y_net = Y_damaged - AbateCost

    Savings = s * Y_damaged
    Consumption = Y_damaged - Savings - AbateCost

    y = (Consumption + AbateCost) / L if L > 0 else 0.0
    y_net = Consumption / L if L > 0 else 0.0

    # Eq 2.1: Potential emissions (unabated)
    Epot = sigma * Y_gross

    # Eq 1.6: Abatement fraction
    if Epot > 0 and AbateCost > 0:
        mu = min(mu_max, (AbateCost * theta2 / (Epot * theta1)) ** (1 / theta2))
    else:
        mu = 0.0

    # Eq 3.5: Mean utility (simplified for Gini = 0)
    if y_net > 0:
        if np.abs(eta - 1.0) < EPSILON:
            U = np.log(y_net)
        else:
            U = (y_net ** (1 - eta)) / (1 - eta)
    else:
        U = NEG_BIGNUM

    # Eq 2.3: Actual emissions (after abatement)
    E = sigma * (1 - mu) * Y_gross

    # Eq 1.10: Capital tendency
    dK_dt = s * Y_net - delta * K

#==========================================================================================
#==========================================================================================
#==========================================================================================

    # Special case: Gini = 0 for DICE-like behavior (no inequality, no regressive damage)
    if not income_dependent_damage_distribution:
        # Simplified calculation without Gini-dependent damage
        Omega = omega_base
    else:
        pareto_scale = pareto_integral_scipy(y_gross, a, c_scale)
        Omega = omega_base * 

        Gini_climate = 0.0


        y = (Consumption + AbateCost) / L if L > 0 else 0.0
        y_net = Consumption / L if L > 0 else 0.0

        G_eff = 0.0
        Redistribution = 0.0
        redistribution = 0.0
        n_iterations = 0



        # Gini dynamics (stays at zero for DICE-like mode)
        d_delta_Gini_dt = 0.0
        delta_Gini_step_change = 0.0

    # Iteratively solve for y_net since climate damage depends on effective income
    elif y_gross > 0 and omega_base < 1.0:
        # Initial guess: analytical approximation
        # Guard against s approaching 1.0 (100% savings, zero consumption)
        if s >= 1.0 - EPSILON:
            # When savings rate is 100%, consumption is zero
            y_net = EPSILON
        else:
            y_half = params['y_damage_distribution_scale']
            omega_approx = omega_base * y_half /( y_gross *(1.0 - s))
            lambda_approx = f * fract_gdp
            y_net = y_gross * (1.0 - omega_approx) * (1-lambda_approx) * (1.0 - s)

        n_iterations = 0
        converged = False
        y_net_prev_prev = None
        y_net_new_prev = None

        while n_iterations < MAX_INITIAL_CAPITAL_ITERATIONS and not converged:
            y_net_prev = y_net
            n_iterations += 1

            # Ensure y_net_prev is positive (negative income is economically impossible)
            # If optimization explores bad parameter space, clamp to small positive value
            # Also guard against nan/inf from numerical instabilities
            if not np.isfinite(y_net_prev) or y_net_prev <= 0:
                y_net_for_damage = EPSILON
            else:
                y_net_for_damage = y_net_prev

            # Calculate climate damage using current income estimate
            # Uses params: psi1, psi2, y_damage_distribution_scale
            Omega, Gini_climate = calculate_climate_damage_and_gini_effect(
                delta_T, Gini, y_net_for_damage, params
            )

            # Clamp Gini_climate to valid bounds (only if not zero)
            if Gini_climate == 0.0:
                pass  # Keep at zero
            else:
                Gini_climate = np.clip(Gini_climate, EPSILON, 1.0 - EPSILON)

            # Eq 1.3: Production after climate damage
            Climate_Damage = Omega * Y_gross
            Y_damaged = Y_gross - Climate_Damage

            # Eq 1.4: Mean per-capita income (after climate damage, before abatement)
            Savings = s * Y_damaged

            # Lambda, fraction of GDP allocated to abatement
            Lambda = f * fract_gdp

            # Eq 1.5: Abatement cost (what society allocates to abatement)
            AbateCost = Lambda * Y_damaged

            # Eq 1.8: Net production after abatement costs
            Y_net = Y_damaged - AbateCost

            # Redistribution amount
            if Gini_climate == 0.0:
                # No inequality, no redistribution needed
                G_eff = 0.0
                Redistribution = 0.0
            elif fract_gdp < 1.0: # do normal redistribution calculation
            # you can't redistribute more than is needed to produce a zero Gini index
                redist_max_fraction = ((2 * Gini_climate) / (1 + Gini_climate)) *                  \
                            ((1 - Gini_climate) / (1 + Gini_climate))**((1-Gini_climate)/ (2*Gini_climate))
                fmin_redist = 1.0 - redist_max_fraction / fract_gdp if fract_gdp > 0 else 0.0
                Redistribution = (1 - max(f,fmin_redist)) * fract_gdp * Y_damaged

                # Eq 4.4:0ibution operates on the climate-damaged distribution
                G_eff, _ = calculate_Gini_effective_redistribute_abate(max(f,fmin_redist), fract_gdp, Gini_climate)
            else: # fract_gdp >= 1, no redistribution
                G_eff = Gini_climate
                Redistribution = 0.0

            # Consumption is the remaining income after savings and abatement costs
            Consumption = Y_damaged - Savings - AbateCost

            # Per-capita income after climate damage and before abatement costs
            y = (Consumption + AbateCost) / L

            # Eq 1.9: Effective per-capita income after climate damage and abatement costs
            # Use Aitken's delta-squared acceleration for faster convergence
            y_net_new = Consumption / L

            if n_iterations == 1:
                # First iteration: use simple update
                y_net = y_net_new
            else:
                # Aitken acceleration: use last two iterations to extrapolate
                delta1 = y_net_prev - y_net_prev_prev
                delta2 = y_net_new - y_net_prev
                denominator = delta2 - delta1

                if np.abs(denominator) > EPSILON:
                    # Apply Aitken's formula with error suppression
                    # (overflow/underflow can occur in extreme parameter space)
                    with np.errstate(all='ignore'):
                        y_net_aitken = y_net_prev_prev - delta1**2 / denominator
                    # Verify Aitken result is valid (positive, finite)
                    if np.isfinite(y_net_aitken) and y_net_aitken > 0:
                        y_net = y_net_aitken
                    else:
                        # Aitken produced invalid result, use simple update
                        y_net = y_net_new
                else:
                    # Denominator too small, use simple update
                    y_net = y_net_new

            # Store values for next iteration
            y_net_prev_prev = y_net_prev
            y_net_new_prev = y_net_new

            # Check convergence using LOOSE_EPSILON for practical precision
            converged = np.abs(y_net - y_net_prev) < LOOSE_EPSILON

        # Final safety check: ensure y_net is valid after iterative solver
        if not np.isfinite(y_net) or y_net <= 0:
            y_net = EPSILON

        # Eq 4.3: Per-capita amount redistributed
        redistribution = Redistribution / L

        # Eq 2.1: Potential emissions (unabated)
        # Note that this implies that you have emissions even for potential output lost to climate damage
        Epot = sigma * Y_gross

        # Eq 1.6: Abatement fraction
        # Note that if the calculated mu exceeds mu_max, and it is cropped to mu_max,
        # then it is just money wasted and the optimizer should do better.
        if Epot > 0 and AbateCost > 0:
            mu = min(mu_max, (AbateCost * theta2 / (Epot * theta1)) ** (1 / theta2))
        else:
            mu = 0.0

        # Eq 3.5: Mean utility
        if y_net > 0 and 0 <= G_eff <= 1.0:
            if np.abs(eta - 1.0) < EPSILON:
                U = np.log(y_net) + np.log((1 - G_eff) / (1 + G_eff)) + 2 * G_eff / (1 + G_eff)
            else:
                term1 = (y_net ** (1 - eta)) / (1 - eta)
                numerator = ((1 + G_eff) ** eta) * ((1 - G_eff) ** (1 - eta))
                denominator = 1 + G_eff * (2 * eta - 1)
                U = term1 * (numerator / denominator)
        else:
            U = NEG_BIGNUM

        # Eq 2.3: Actual emissions (after abatement)
        E = sigma * (1 - mu) * Y_gross

        # Eq 1.10: Capital tendency
        dK_dt = s * Y_net - delta * K

        # Gini perturbation dynamics (restore to zero)
        d_delta_Gini_dt = -Gini_restore * delta_Gini
        delta_Gini_step_change = Gini_fract * (G_eff - Gini)

    else:
        Omega = 0.0
        Gini_climate = Gini
        Climate_Damage = 0.0
        Y_damaged = 0.0
        Savings = 0.0
        Lambda = 0.0
        AbateCost = 0.0
        Y_net = 0.0
        Redistribution = 0.0
        Consumption = 0.0
        y = 0.0
        y_net = 0.0
        redistribution = 0.0
        n_iterations = 0
        G_eff = Gini
        U = NEG_BIGNUM
        E = 0.0
        dK_dt = -delta * K
        d_delta_Gini_dt = -Gini_restore * delta_Gini
        delta_Gini_step_change = Gini_fract * (G_eff - Gini)
        
    # Prepare output
    results = {}

    if store_detailed_output:
        # Additional calculated variables for detailed output only
        marginal_abatement_cost = theta1 * mu ** (theta2 - 1)  # Social cost of carbon
        Consumption = y * L  # Total Consumption
        discounted_utility = U * np.exp(-rho * t)  # Discounted utility

        # Return full diagnostics for CSV/PDF output
        results.update({
            'dK_dt': dK_dt,
            'dEcum_dt': E,
            'd_delta_Gini_dt': d_delta_Gini_dt,
            'delta_Gini_step_change': delta_Gini_step_change,
            'Gini': Gini,  # Total Gini (background + perturbation) for plotting
            'Gini_background': Gini_background,  # Background Gini for reference
            'Y_gross': Y_gross,
            'delta_T': delta_T,
            'Omega': Omega,
            'Gini_climate': Gini_climate,
            'Y_damaged': Y_damaged,
            'Y_net': Y_net,
            'y': y,
            'redistribution': redistribution,
            'mu': mu,
            'Lambda': Lambda,
            'AbateCost': AbateCost,
            'marginal_abatement_cost': marginal_abatement_cost,
            'y_net': y_net,
            'G_eff': G_eff,
            'U': U,
            'E': E,
            'Climate_Damage': Climate_Damage,
            'Savings': Savings,
            'Consumption': Consumption,
            'discounted_utility': discounted_utility,
            's': s,  # Savings rate (currently constant, may become time-dependent)
            'n_iterations': n_iterations,  # Number of iterations for climate damage convergence
        })

        # Return minimal variables needed for optimization
    results.update( {
        'U': U,
        'dK_dt': dK_dt,
        'dEcum_dt': E,
        'd_delta_Gini_dt': d_delta_Gini_dt,
        'delta_Gini_step_change': delta_Gini_step_change,
    })
    
    return results


def integrate_model(config, store_detailed_output=True):
    """
    Integrate the model forward in time using Euler's method.

    Parameters
    ----------
    config : ModelConfiguration
        Complete model configuration including parameters and time-dependent functions
    store_detailed_output : bool, optional
        If True (default), stores all diagnostic variables for CSV/PDF output.
        If False, stores only t, U needed for optimization objective calculation.

    Returns
    -------
    dict
        Time series results with keys:
        - 't': array of time points
        - 'U': array of utility values (always stored)
        - 'L': array of population values (always stored, needed for objective function)

        If store_detailed_output=True, also includes:
        - 'K': array of capital stock values
        - 'Ecum': array of cumulative emissions values
        - 'delta_Gini': array of Gini perturbation values
        - 'Gini': array of total Gini index values (background + perturbation)
        - 'Gini_background': array of background Gini index values
        - 'A', 'sigma', 'theta1', 'f': time-dependent inputs
        - All derived variables: Y_gross, delta_T, Omega, Gini_climate, Y_damaged, Y_net,
          y, redistribution, mu, Lambda, AbateCost, marginal_abatement_cost, y_net, G_eff, E
        - 'd_delta_Gini_dt', 'delta_Gini_step_change': perturbation tendencies

    Notes
    -----
    Uses simple Euler integration: state(t+dt) = state(t) + dt * tendency(t)
    This ensures all functional relationships are satisfied exactly at output points.

    Initial conditions are computed automatically:
    - Ecum(0) = Ecum_initial (initial cumulative emissions from configuration)
    - K(0) = (s·A(0)/δ)^(1/(1-α))·L(0) (steady-state capital)
    - Gini(0) = Gini_background(t_start) (background Gini from time function at start)
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
    delta = config.scalar_params.delta
    alpha = config.scalar_params.alpha
    fract_gdp = config.scalar_params.fract_gdp

    # take abatement cost and initial climate damage into account for initial capital
    Ecum_initial = config.scalar_params.Ecum_initial
    params = evaluate_params_at_time(t_start, config)

    Gini = config.time_functions['Gini_background'](t_start)
    k_climate = params['k_climate']
    delta_T = k_climate * Ecum_initial

    # iterate to find K0 that is consistent with climate damage from initial emissions
    Omega_prev = 1.0
    Omega_current = 0.0
    n_iterations = 0

    """
    # get time-dependent parameters at t_start
    s0 = params['s']
    f0 = params['f']
    k_climate = params['k_climate']
    lambda0 = (1-s0) * f0 * fract_gdp

    while np.abs(Omega_current - Omega_prev) > EPSILON:
        n_iterations += 1
        if n_iterations > MAX_INITIAL_CAPITAL_ITERATIONS:
            raise RuntimeError(
                f"Initial capital stock failed to converge after {MAX_INITIAL_CAPITAL_ITERATIONS} iterations. "
                f"Omega_prev = {Omega_prev:.10f}, Omega_current = {Omega_current:.10f}, "
                f"difference = {np.abs(Omega_current - Omega_prev):.2e} (tolerance: {EPSILON:.2e})"
            )
        Omega_prev = Omega_current
        K0 = ((s0 * (1 - Omega_prev) * (1 - lambda0) * A0 / delta) ** (1 / (1 - alpha))) * L0
        y_gross = A0 * (K0 ** alpha) * (L0 ** (1 - alpha)) / L0
        Omega_current, _ = calculate_climate_damage_and_gini_effect(
            delta_T, Gini, y_gross, params
        )

    """
    state = {
        'K': config.scalar_params.K_initial,
        'Ecum': config.scalar_params.Ecum_initial,
        'delta_Gini': 0.0  # Initialize perturbation to zero
    }

    # Initialize storage for variables
    results = {}

    if store_detailed_output:
        # Add storage for all diagnostic variables
        results.update({
            'A': np.zeros(n_steps),
            'sigma': np.zeros(n_steps),
            'theta1': np.zeros(n_steps),
            'f': np.zeros(n_steps),
            'Y_gross': np.zeros(n_steps),
            'delta_T': np.zeros(n_steps),
            'Omega': np.zeros(n_steps),
            'Gini': np.zeros(n_steps),  # Total Gini (background + perturbation)
            'Gini_background': np.zeros(n_steps),  # Background Gini
            'Gini_climate': np.zeros(n_steps),
            'Y_damaged': np.zeros(n_steps),
            'Y_net': np.zeros(n_steps),
            'y': np.zeros(n_steps),
            'redistribution': np.zeros(n_steps),
            'mu': np.zeros(n_steps),
            'Lambda': np.zeros(n_steps),
            'AbateCost': np.zeros(n_steps),
            'marginal_abatement_cost': np.zeros(n_steps),
            'y_net': np.zeros(n_steps),
            'G_eff': np.zeros(n_steps),
            'E': np.zeros(n_steps),
            'dK_dt': np.zeros(n_steps),
            'dEcum_dt': np.zeros(n_steps),
            'd_delta_Gini_dt': np.zeros(n_steps),
            'delta_Gini_step_change': np.zeros(n_steps),
            'Climate_Damage': np.zeros(n_steps),
            'Savings': np.zeros(n_steps),
            'Consumption': np.zeros(n_steps),
            'discounted_utility': np.zeros(n_steps),
            's': np.zeros(n_steps),
            'n_iterations': np.zeros(n_steps, dtype=int),
        })

    # Always store time, state variables, and objective function variables
    results.update({
        't': t_array,
        'K': np.zeros(n_steps),
        'Ecum': np.zeros(n_steps),
        'delta_Gini': np.zeros(n_steps),
        'U': np.zeros(n_steps),
        'L': np.zeros(n_steps),  # Needed for objective function
    })

    # Time stepping loop
    for i, t in enumerate(t_array):
        # Evaluate time-dependent parameters at current time
        params = evaluate_params_at_time(t, config)

        # Calculate all variables and tendencies at current time
        outputs = calculate_tendencies(state, params, store_detailed_output)

        # Always store variables needed for objective function
        results['U'][i] = outputs['U']
        results['L'][i] = params['L']

        if store_detailed_output:
            # Store state variables
            results['K'][i] = state['K']
            results['Ecum'][i] = state['Ecum']
            results['delta_Gini'][i] = state['delta_Gini']

            # Store time-dependent inputs
            results['A'][i] = params['A']
            results['sigma'][i] = params['sigma']
            results['theta1'][i] = params['theta1']
            results['f'][i] = params['f']

            # Store all derived variables
            results['Y_gross'][i] = outputs['Y_gross']
            results['delta_T'][i] = outputs['delta_T']
            results['Omega'][i] = outputs['Omega']
            results['Gini'][i] = outputs['Gini']  # Total Gini
            results['Gini_background'][i] = outputs['Gini_background']  # Background Gini
            results['Gini_climate'][i] = outputs['Gini_climate']
            results['Y_damaged'][i] = outputs['Y_damaged']
            results['Y_net'][i] = outputs['Y_net']
            results['y'][i] = outputs['y']
            results['redistribution'][i] = outputs['redistribution']
            results['mu'][i] = outputs['mu']
            results['Lambda'][i] = outputs['Lambda']
            results['AbateCost'][i] = outputs['AbateCost']
            results['marginal_abatement_cost'][i] = outputs['marginal_abatement_cost']
            results['y_net'][i] = outputs['y_net']
            results['G_eff'][i] = outputs['G_eff']
            results['E'][i] = outputs['E']
            results['dK_dt'][i] = outputs['dK_dt']
            results['dEcum_dt'][i] = outputs['dEcum_dt']
            results['d_delta_Gini_dt'][i] = outputs['d_delta_Gini_dt']
            results['delta_Gini_step_change'][i] = outputs['delta_Gini_step_change']
            results['Climate_Damage'][i] = outputs['Climate_Damage']
            results['Savings'][i] = outputs['Savings']
            results['Consumption'][i] = outputs['Consumption']
            results['discounted_utility'][i] = outputs['discounted_utility']
            results['s'][i] = outputs['s']
            results['n_iterations'][i] = outputs['n_iterations']

        # Euler step: update state for next iteration (skip on last step)
        if i < n_steps - 1:
            state['K'] = state['K'] + dt * outputs['dK_dt']
            # do not allow cumulative emissions to go negative, making it colder than the initial condition
            state['Ecum'] = max(0.0, state['Ecum'] + dt * outputs['dEcum_dt'])
            # delta_Gini update includes both continuous change and discontinuous step
            state['delta_Gini'] = state['delta_Gini'] + dt * outputs['d_delta_Gini_dt'] + outputs['delta_Gini_step_change']

    return results
