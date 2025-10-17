"""
Parameter definitions and configurations for COIN_equality model.

This module provides three categories of parameters:
1. Scalar parameters - time-invariant model constants
2. Time-dependent functions - evaluated at specific times
3. Integration/optimization control parameters

All parameters are required (no defaults) following the fail-fast philosophy.

Configuration is loaded from a JSON file that specifies all parameter values
and the run name (used for output directory naming).
"""

import json
import numpy as np
from dataclasses import dataclass


# =============================================================================
# Time-Dependent Function Factories
# =============================================================================

def create_constant(value):
    """
    Create a constant function that returns the same value for all times.

    Parameters
    ----------
    value : float
        The constant value to return

    Returns
    -------
    callable
        Function f(t) = value
    """
    return lambda t: value


def create_exponential_growth(initial_value, growth_rate):
    """
    Create an exponential growth/decay function.

    Parameters
    ----------
    initial_value : float
        Value at t=0
    growth_rate : float
        Growth rate (yr^-1). Positive for growth, negative for decay.

    Returns
    -------
    callable
        Function f(t) = initial_value * exp(growth_rate * t)

    Examples
    --------
    Population growth at 1% per year:
    >>> L = create_exponential_growth(7e9, 0.01)

    Carbon intensity declining at 2% per year:
    >>> sigma = create_exponential_growth(0.5, -0.02)
    """
    return lambda t: initial_value * np.exp(growth_rate * t)


def create_logistic_growth(L0, L_inf, growth_rate):
    """
    Create a logistic (S-curve) growth function.

    Parameters
    ----------
    L0 : float
        Initial value at t=0
    L_inf : float
        Asymptotic limit as t -> infinity
    growth_rate : float
        Intrinsic growth rate (yr^-1)

    Returns
    -------
    callable
        Function f(t) = L_inf / (1 + ((L_inf/L0) - 1) * exp(-growth_rate * t))

    Examples
    --------
    Population growth from 7B to 10B:
    >>> L = create_logistic_growth(7e9, 10e9, 0.02)
    """
    return lambda t: L_inf / (1 + ((L_inf / L0) - 1) * np.exp(-growth_rate * t))


def create_piecewise_linear(time_points, values):
    """
    Create a piecewise linear function from discrete points.

    Parameters
    ----------
    time_points : array_like
        Time values (must be monotonically increasing)
    values : array_like
        Function values at each time point

    Returns
    -------
    callable
        Function that linearly interpolates between points

    Examples
    --------
    Abatement cost declining in steps:
    >>> theta1 = create_piecewise_linear([0, 50, 100], [0.1, 0.05, 0.02])
    """
    time_points = np.asarray(time_points)
    values = np.asarray(values)
    return lambda t: np.interp(t, time_points, values)


# =============================================================================
# Control Function Factories
# =============================================================================

def constant_control(f_value):
    """
    Create a constant control function.

    Parameters
    ----------
    f_value : float
        Constant fraction allocated to abatement (0 <= f <= 1)

    Returns
    -------
    callable
        Function f(t) = f_value
    """
    return lambda t: f_value


def piecewise_constant_control(time_points, f_values):
    """
    Create a piecewise constant control function for optimization.

    Parameters
    ----------
    time_points : array_like
        Time boundaries for each constant segment
    f_values : array_like
        Control values for each segment

    Returns
    -------
    callable
        Function that returns appropriate f_value for time t

    Notes
    -----
    This is the typical discretization for direct optimization methods.
    The optimizer adjusts the f_values vector.
    """
    time_points = np.asarray(time_points)
    f_values = np.asarray(f_values)
    return lambda t: f_values[np.searchsorted(time_points[1:], t)]


# =============================================================================
# Parameter Dataclasses
# =============================================================================

@dataclass
class ScalarParameters:
    """
    Time-invariant scalar parameters for the economic-climate model.

    All parameters are required (no defaults).

    Attributes
    ----------
    alpha : float
        Output elasticity of capital (capital share of income)
    delta : float
        Capital depreciation rate (yr^-1)
    s : float
        Savings rate (fraction of net production)
    k_damage : float
        Climate damage coefficient (°C^-beta)
    beta : float
        Climate damage exponent
    k_climate : float
        Temperature sensitivity to cumulative emissions (°C tCO2^-1)
    eta : float
        Coefficient of relative risk aversion (CRRA utility parameter)
    rho : float
        Pure rate of time preference (yr^-1)
    G1 : float
        Initial Gini index (0 < G1 < 1)
    deltaL : float
        Fraction of income available for redistribution
    theta2 : float
        Abatement cost exponent
    """
    alpha: float
    delta: float
    s: float
    k_damage: float
    beta: float
    k_climate: float
    eta: float
    rho: float
    G1: float
    deltaL: float
    theta2: float


@dataclass
class IntegrationParameters:
    """
    Parameters controlling time integration and optimization.

    Attributes
    ----------
    t_start : float
        Start time for integration (yr)
    t_end : float
        End time for integration (yr)
    dt : float
        Time step for Euler integration (yr)
    rtol : float
        Relative tolerance for ODE solver (reserved for future use)
    atol : float
        Absolute tolerance for ODE solver (reserved for future use)
    """
    t_start: float
    t_end: float
    dt: float
    rtol: float
    atol: float


@dataclass
class ModelConfiguration:
    """
    Complete model configuration bundling all parameters.

    Attributes
    ----------
    run_name : str
        Name for this run (used for output directory naming)
    scalar_params : ScalarParameters
        Time-invariant scalar parameters
    time_functions : dict
        Dictionary mapping parameter names to time-dependent callables.
        Required keys: 'A', 'L', 'sigma', 'theta1'
    integration_params : IntegrationParameters
        Integration and optimization control parameters
    initial_state : dict
        Initial conditions. Required keys: 'K', 'Ecum'
    control_function : callable
        Control function f(t) returning fraction allocated to abatement
    """
    run_name: str
    scalar_params: ScalarParameters
    time_functions: dict
    integration_params: IntegrationParameters
    initial_state: dict
    control_function: callable


# =============================================================================
# Parameter Evaluation
# =============================================================================

def evaluate_params_at_time(t, config):
    """
    Evaluate all parameters at a specific time.

    Combines scalar parameters with time-dependent function evaluations
    into a single dictionary suitable for use with calculate_tendencies().

    Parameters
    ----------
    t : float
        Time at which to evaluate parameters (yr)
    config : ModelConfiguration
        Complete model configuration

    Returns
    -------
    dict
        Dictionary containing all parameters evaluated at time t,
        with keys matching those expected by calculate_tendencies():
        'alpha', 'delta', 's', 'k_damage', 'beta', 'k_climate',
        'eta', 'rho', 'G1', 'deltaL', 'theta2',
        'A', 'L', 'sigma', 'theta1', 'f'
    """
    sp = config.scalar_params
    tf = config.time_functions

    return {
        # Scalar parameters
        'alpha': sp.alpha,
        'delta': sp.delta,
        's': sp.s,
        'k_damage': sp.k_damage,
        'beta': sp.beta,
        'k_climate': sp.k_climate,
        'eta': sp.eta,
        'rho': sp.rho,
        'G1': sp.G1,
        'delta_L': sp.deltaL,
        'theta2': sp.theta2,

        # Time-dependent function evaluations
        'A': tf['A'](t),
        'L': tf['L'](t),
        'sigma': tf['sigma'](t),
        'theta1': tf['theta1'](t),

        # Control function evaluation
        'f': config.control_function(t),
    }


# =============================================================================
# Configuration Loading from JSON
# =============================================================================

def _create_time_function(func_spec):
    """
    Create a time-dependent function from JSON specification.

    Parameters
    ----------
    func_spec : dict
        Dictionary with 'type' key and type-specific parameters

    Returns
    -------
    callable
        Time-dependent function
    """
    func_type = func_spec['type']

    if func_type == 'constant':
        return create_constant(func_spec['value'])
    elif func_type == 'exponential_growth':
        return create_exponential_growth(
            func_spec['initial_value'],
            func_spec['growth_rate']
        )
    elif func_type == 'logistic_growth':
        return create_logistic_growth(
            func_spec['L0'],
            func_spec['L_inf'],
            func_spec['growth_rate']
        )
    elif func_type == 'piecewise_linear':
        return create_piecewise_linear(
            func_spec['time_points'],
            func_spec['values']
        )


def _create_control_function(control_spec):
    """
    Create a control function from JSON specification.

    Parameters
    ----------
    control_spec : dict
        Dictionary with 'type' key and type-specific parameters

    Returns
    -------
    callable
        Control function f(t)
    """
    control_type = control_spec['type']

    if control_type == 'constant':
        return constant_control(control_spec['value'])
    elif control_type == 'piecewise_constant':
        return piecewise_constant_control(
            control_spec['time_points'],
            control_spec['values']
        )


def calculate_initial_capital(s, A0, L0, delta, alpha):
    """
    Calculate steady-state initial capital stock.

    Parameters
    ----------
    s : float
        Savings rate
    A0 : float
        Total factor productivity at t=0
    L0 : float
        Population at t=0
    delta : float
        Capital depreciation rate (yr^-1)
    alpha : float
        Output elasticity of capital

    Returns
    -------
    float
        Initial capital stock at steady state with no climate damage or abatement

    Notes
    -----
    At steady state with dK/dt = 0, no climate damage (Ω=0), and no abatement (Λ=0):
        s · A · K^α · L^(1-α) = δ · K

    Solving for K:
        K = (s · A / δ)^(1/(1-α)) · L
    """
    return ((s * A0 / delta) ** (1 / (1 - alpha))) * L0


def load_configuration(config_path):
    """
    Load model configuration from JSON file.

    Parameters
    ----------
    config_path : str
        Path to JSON configuration file

    Returns
    -------
    ModelConfiguration
        Complete model configuration loaded from file

    Notes
    -----
    The JSON file must contain:
    - run_name: string identifier for this run
    - scalar_parameters: dict with all ScalarParameters fields
    - time_functions: dict with specs for A, L, sigma, theta1
    - integration_parameters: dict with t_start, t_end, dt, rtol, atol
    - control_function: dict with type and parameters

    Initial state is computed automatically:
    - Ecum(0) = 0 (no cumulative emissions at start)
    - K(0) = steady-state capital with no climate damage or abatement

    See config_baseline.json for an example.
    """
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # Create scalar parameters
    scalar_params = ScalarParameters(**config_data['scalar_parameters'])

    # Create time-dependent functions
    time_functions = {
        name: _create_time_function(spec)
        for name, spec in config_data['time_functions'].items()
    }

    # Create integration parameters
    integration_params = IntegrationParameters(**config_data['integration_parameters'])

    # Create control function
    control_function = _create_control_function(config_data['control_function'])

    # Extract run name
    run_name = config_data['run_name']

    # Calculate initial state automatically
    t0 = integration_params.t_start
    A0 = time_functions['A'](t0)
    L0 = time_functions['L'](t0)

    K0 = calculate_initial_capital(
        s=scalar_params.s,
        A0=A0,
        L0=L0,
        delta=scalar_params.delta,
        alpha=scalar_params.alpha
    )

    initial_state = {
        'K': K0,
        'Ecum': 0.0
    }

    return ModelConfiguration(
        run_name=run_name,
        scalar_params=scalar_params,
        time_functions=time_functions,
        integration_params=integration_params,
        initial_state=initial_state,
        control_function=control_function,
    )
