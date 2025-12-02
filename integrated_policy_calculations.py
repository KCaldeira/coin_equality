"""
Integrated policy calculations for climate-economy model.

This module provides a comprehensive function that calculates economic outcomes
given policy switches, climate state, and previous income distribution. It handles:
- Climate damage (uniform or income-dependent)
- Taxation (uniform fractional or progressive)
- Redistribution (uniform dividend or targeted to lowest income)
- Resulting income distribution and utility

The design uses the previous time step's income distribution to avoid circular
dependencies, as documented in IMPLEMENTATION_PLAN.md Section 4 and 5.
"""

import numpy as np
from income_distribution import a_from_G, G_from_a
from constants import EPSILON


def calculate_integrated_economy(
    delta_T,
    previous_step_values,
    Y_gross,
    L,
    s,
    fract_gdp,
    f,
    eta,
    psi1,
    psi2,
    y_damage_aggregate_scale,
    y_damage_distribution_coeff,
    income_dependent_aggregate_damage,
    income_dependent_damage_distribution,
    income_dependent_tax_policy,
    income_redistribution,
    income_dependent_redistribution_policy,
    n_discrete=1000
):
    """
    Calculate integrated economic outcomes with climate damage, taxation, and redistribution.

    This function performs a complete economic calculation for one time step, using the
    previous time step's income distribution to determine climate damage (avoiding
    circular dependency). It then applies current policy decisions for taxation and
    redistribution.

    Parameters
    ----------
    delta_T : float
        Temperature change above baseline (°C)
    previous_step_values : dict
        Income distribution from previous time step:
        - 'y_mean': Mean income ($)
        - 'gini': Gini coefficient
        - 'F_crit_tax': Critical rank for taxation (optional, only if progressive tax)
        - 'F_crit_redistribution': Critical rank for redistribution (optional, only if targeted)
    Y_gross : float
        Gross production before damage ($)
    L : float
        Population (number of people)
    s : float
        Savings rate (fraction)
    fract_gdp : float
        Fraction of GDP available for taxation/redistribution (0 to 1)
    f : float
        Fraction of fract_gdp allocated to abatement vs redistribution (0 to 1)
    eta : float
        Coefficient of relative risk aversion
    psi1 : float
        Linear climate damage coefficient (°C⁻¹)
    psi2 : float
        Quadratic climate damage coefficient (°C⁻²)
    y_damage_aggregate_scale : float
        Income scale for aggregate damage saturation ($)
    y_damage_distribution_coeff : float
        Damage distribution coefficient parameter
    income_dependent_aggregate_damage : bool
        If True, aggregate damage decreases as world gets richer
    income_dependent_damage_distribution : bool
        If True, damage weighted towards low-income individuals
    income_dependent_tax_policy : bool
        If True, progressive tax (tax richest); if False, uniform fractional tax
    income_redistribution : bool
        If True, enable redistribution; if False, no redistribution (all revenues to abatement)
    income_dependent_redistribution_policy : bool
        If True, targeted to lowest income; if False, uniform dividend
        Only applies when income_redistribution=True
    n_discrete : int, optional
        Number of discrete segments for numerical integration over income distribution
        Default: 1000

    Returns
    -------
    dict
        Dictionary containing:
        - 'Omega': Aggregate damage fraction (0 ≤ Ω < 1)
        - 'Gini_climate': Gini after climate damage
        - 'Y_damaged': Production after climate damage ($)
        - 'Y_net': Net production after abatement ($)
        - 'Consumption': Total consumption ($)
        - 'y_net': Per-capita net income ($)
        - 'G_eff': Effective Gini after all transformations
        - 'U': Mean utility
        - 'F_crit_tax': Critical rank for taxation (for next time step)
        - 'F_crit_redistribution': Critical rank for redistribution (for next time step)
        - 'current_income_dist': Income distribution for next time step

    Notes
    -----
    This function aims to replace/consolidate:
    - calculate_climate_damage_from_prev_distribution()
    - calculate_Gini_effective_redistribute_abate()
    - Parts of calculate_tendencies()
    """
    # Extract previous time step info
    y_mean_prev = previous_step_values['y_mean']
    gini_prev = previous_step_values['gini']

    # Get critical ranks from previous time step if applicable
    if income_dependent_tax_policy and 'F_crit_tax' in previous_step_values:
        F_crit_tax_prev = previous_step_values['F_crit_tax']
    else:
        F_crit_tax_prev = 1.0  # Everyone pays same tax rate

    if income_dependent_redistribution_policy and 'F_crit_redistribution' in previous_step_values:
        F_crit_redistribution_prev = previous_step_values['F_crit_redistribution']
    else:
        F_crit_redistribution_prev = 0.0  # Everyone receives the same redistribution amount

    #===================================================================================
    # STEP 1: Calculate climate damage using previous time step's distribution
    #===================================================================================

    # Calculate base aggregate climate damage
    omega_base = psi1 * delta_T + psi2 * delta_T**2

    # TODO: Implement aggregate damage calculation with income dependence
    # For now, use simple aggregate damage
    Omega = omega_base

    # TODO: Implement Gini_climate calculation with income-dependent damage distribution
    # For now, assume no change to Gini from damage
    Gini_climate = gini_prev

    #===================================================================================
    # STEP 2: Calculate production after damage
    #===================================================================================

    Climate_Damage = Omega * Y_gross
    Y_damaged = Y_gross - Climate_Damage

    #===================================================================================
    # STEP 3: Calculate taxation and redistribution
    #===================================================================================

    # Savings and abatement
    Savings = s * Y_damaged
    Lambda = f * fract_gdp
    AbateCost = Lambda * Y_damaged

    # TODO: Calculate critical ranks for current time step
    # For now, use placeholder values
    F_crit_tax_current = 1.0 if not income_dependent_tax_policy else 0.9
    F_crit_redistribution_current = 0.0 if not income_dependent_redistribution_policy else 0.1

    # TODO: Implement actual taxation and redistribution calculations
    # For now, use simplified calculation
    Redistribution = (1 - f) * fract_gdp * Y_damaged if fract_gdp < 1.0 else 0.0

    #===================================================================================
    # STEP 4: Calculate effective Gini after all transformations
    #===================================================================================

    # TODO: Replace with proper calculation accounting for all policies
    # For now, use simplified approximation
    G_eff = Gini_climate

    #===================================================================================
    # STEP 5: Calculate consumption and utility
    #===================================================================================

    Y_net = Y_damaged - AbateCost
    Consumption = Y_damaged - Savings - AbateCost

    y_net = Consumption / L if L > 0 else 0.0

    # Calculate utility
    if y_net > 0 and 0 <= G_eff <= 1.0:
        if np.abs(eta - 1.0) < EPSILON:
            U = np.log(y_net) + np.log((1 - G_eff) / (1 + G_eff)) + 2 * G_eff / (1 + G_eff)
        else:
            term1 = (y_net ** (1 - eta)) / (1 - eta)
            numerator = ((1 + G_eff) ** eta) * ((1 - G_eff) ** (1 - eta))
            denominator = 1 + G_eff * (2 * eta - 1)
            U = term1 * (numerator / denominator)
    else:
        from constants import NEG_BIGNUM
        U = NEG_BIGNUM

    #===================================================================================
    # STEP 6: Prepare output
    #===================================================================================

    # Income distribution for next time step
    current_income_dist = {
        'y_mean': y_net,
        'gini': G_eff,
    }

    if income_dependent_tax_policy:
        current_income_dist['F_crit_tax'] = F_crit_tax_current

    if income_dependent_redistribution_policy:
        current_income_dist['F_crit_redistribution'] = F_crit_redistribution_current

    return {
        'Omega': Omega,
        'Gini_climate': Gini_climate,
        'Y_damaged': Y_damaged,
        'Y_net': Y_net,
        'Consumption': Consumption,
        'y_net': y_net,
        'G_eff': G_eff,
        'U': U,
        'F_crit_tax': F_crit_tax_current,
        'F_crit_redistribution': F_crit_redistribution_current,
        'current_income_dist': current_income_dist,
        'Climate_Damage': Climate_Damage,
        'Savings': Savings,
        'AbateCost': AbateCost,
        'Redistribution': Redistribution,
    }
