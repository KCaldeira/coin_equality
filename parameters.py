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
from constants import LOOSE_EPSILON


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


def create_double_exponential_growth(initial_value, growth_rate_1, growth_rate_2, fract_1):
    """
    Create a double exponential growth function (Barrage & Nordhaus 2023).

    Parameters
    ----------
    initial_value : float
        Value at t=0
    growth_rate_1 : float
        First growth rate (yr^-1)
    growth_rate_2 : float
        Second growth rate (yr^-1)
    fract_1 : float
        Fraction assigned to first exponential (0 ≤ fract_1 ≤ 1)

    Returns
    -------
    callable
        Function f(t) = initial_value * (fract_1 * exp(growth_rate_1 * t) + (1 - fract_1) * exp(growth_rate_2 * t))

    Examples
    --------
    Carbon intensity with two-phase decline:
    >>> sigma = create_double_exponential_growth(0.5, -0.02, -0.005, 0.7)
    """
    return lambda t: initial_value * (fract_1 * np.exp(growth_rate_1 * t) + (1 - fract_1) * np.exp(growth_rate_2 * t))


def create_gompertz_growth(initial_value, final_value, adjustment_coefficient):
    """
    Create a Gompertz growth function.

    Parameters
    ----------
    initial_value : float
        Value at t=0
    final_value : float
        Asymptotic limit as t → ∞ (for adjustment_coefficient < 0)
    adjustment_coefficient : float
        Growth rate parameter (yr^-1). Typically negative for growth from initial to final value.

    Returns
    -------
    callable
        Function L(t) = final_value * exp(ln((initial_value / final_value)) * exp(adjustment_coefficient * t)

    Notes
    -----
    At t=0: L(0) = initial_value
    As t → ∞ (with adjustment_coefficient < 0): L(t) → final_value

    Examples
    --------
    Population growing from 7B to 10B:
    >>> L = create_gompertz_growth(7e9, 10e9, -0.02)
    """
    return lambda t: final_value * np.exp(np.log(initial_value / final_value) * np.exp(adjustment_coefficient * t))


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
# Control Functions for f and s (for simultaneous optimization of f and s)
# =============================================================================

def create_f_and_s_control_from_single(f_control, s_time_function):
    """
    Create a control function for both f and s from separate f control and s time function.

    This is used during Phase 2 transition to allow f to be optimized while
    s follows a prescribed time function.

    Parameters
    ----------
    f_control : callable
        Single control function f(t) returning scalar
    s_time_function : callable
        Time function s(t) returning scalar

    Returns
    -------
    callable
        Control function returning tuple (f, s)
    """
    return lambda t: (f_control(t), s_time_function(t))


def create_f_and_s_control_from_specs(f_spec, s_spec):
    """
    Create a control function for both f and s from separate specifications for f and s.

    Parameters
    ----------
    f_spec : dict
        Control specification for f (same format as single control function)
    s_spec : dict
        Control specification for s (same format as time functions)

    Returns
    -------
    callable
        Control function returning tuple (f, s)
    """
    f_control = _create_time_function(f_spec) if isinstance(f_spec, dict) else f_spec
    s_control = _create_time_function(s_spec) if isinstance(s_spec, dict) else s_spec
    return lambda t: (f_control(t), s_control(t))


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
    psi1 : float
        Linear climate damage coefficient (°C⁻¹) [Barrage & Nordhaus 2023]
    psi2 : float
        Quadratic climate damage coefficient (°C⁻²) [Barrage & Nordhaus 2023]
    y_damage_halfsat : float
        Income half-saturation for climate damage ($)
        (income level at which damage is 50% of maximum; lower = more regressive)
    k_climate : float
        Temperature sensitivity to cumulative emissions (°C tCO2^-1)
    eta : float
        Coefficient of relative risk aversion (CRRA utility parameter)
    rho : float
        Pure rate of time preference (yr^-1)
    Gini_initial : float
        Initial Gini index (0 < Gini_initial < 1)
    Gini_fract : float
        Fraction of effective Gini change as instantaneous step (0 <= Gini_fract <= 1)
    Gini_restore : float
        Rate at which Gini restores to initial value (yr^-1, 0=no restoration)
    fract_gdp : float
        Fraction of GDP available for redistribution and abatement
    theta2 : float
        Abatement cost exponent
    mu_max : float
        Maximum allowed abatement fraction (cap on μ)
        Default: INVERSE_EPSILON (effectively no cap)
    Ecum_initial : float
        Initial cumulative CO2 emissions (tCO2)
        Default: 0.0 (no prior emissions)
    K_initial : float
        Initial capital stock ($)
        Required parameter, must be specified in configuration
    """
    alpha: float
    delta: float
    psi1: float
    psi2: float
    y_damage_halfsat: float
    k_climate: float
    eta: float
    rho: float
    Gini_initial: float
    Gini_fract: float
    Gini_restore: float
    fract_gdp: float
    theta2: float
    K_initial: float
    mu_max: float = None  # Will be set to INVERSE_EPSILON in __post_init__ if None
    Ecum_initial: float = 0.0  # Default to zero (no prior emissions)

    def __post_init__(self):
        """Set default for mu_max if not provided."""
        from constants import INVERSE_EPSILON
        if self.mu_max is None:
            object.__setattr__(self, 'mu_max', INVERSE_EPSILON)


@dataclass
class IntegrationParameters:
    """
    Parameters controlling time integration.

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
    plot_short_horizon : float
        Time horizon (yr) for short-term detailed plots.
        Creates second PDF with results from t_start to plot_short_horizon.
    """
    t_start: float
    t_end: float
    dt: float
    rtol: float
    atol: float
    plot_short_horizon: float


@dataclass
class OptimizationParameters:
    """
    Parameters controlling optimization.

    Supports two modes:
    1. Direct multi-point optimization (control_times is list, initial_guess_f is list)
    2. Iterative refinement optimization (control_times is int, initial_guess_f is float)

    Supports dual optimization when initial_guess_s is specified.

    Attributes
    ----------
    max_evaluations : int
        Maximum number of objective function evaluations (per iteration for iterative mode)
    control_times : list of float OR int
        Direct mode (list): Times (years) where control points are placed.
            For single-point: [0]
            For multi-point: e.g., [0, 25, 50, 75, 100]
        Iterative refinement mode (int): Number of refinement iterations.
            Iteration 1: 2 control points (t_start, t_end)
            Iteration 2: 3 control points
            Iteration k: 2^k + 1 control points
        Note: Both f and s use the same control_times in dual optimization mode.
    initial_guess_f : list of float OR float
        Direct mode (list): Initial f values at each control time.
            Must have same length as control_times.
            Each value must satisfy 0 ≤ f ≤ 1.
        Iterative refinement mode (float): Single initial f value for first iteration.
            Must satisfy 0 ≤ f ≤ 1.
    algorithm : str, optional
        NLopt algorithm to use. If None, defaults to 'LN_SBPLX'.
        Options include:
        - 'LN_SBPLX': Local derivative-free Subplex (default, robust for noisy objectives)
        - 'LN_BOBYQA': Local derivative-free (good for smooth problems)
        - 'GN_ISRES': Global stochastic (good for multi-modal problems)
        - 'GN_DIRECT_L': Global deterministic (good for Lipschitz-continuous)
        - 'LN_COBYLA': Local derivative-free (handles nonlinear constraints)
        - 'LN_NELDERMEAD': Local derivative-free (Nelder-Mead simplex)
        See NLopt documentation for full list.
    ftol_rel : float, optional
        Relative tolerance on objective function changes.
        Stops when |Δf| < ftol_rel * |f|.
        If None, uses NLopt default (0.0 = disabled).
    ftol_abs : float, optional
        Absolute tolerance on objective function changes.
        Stops when |Δf| < ftol_abs.
        If None, uses NLopt default (0.0 = disabled).
    xtol_rel : float, optional
        Relative tolerance on parameter changes.
        Stops when |Δx| < xtol_rel * |x| for all parameters.
        If None, uses NLopt default (0.0 = disabled).
    xtol_abs : float, optional
        Absolute tolerance on parameter changes.
        Stops when |Δx| < xtol_abs for all parameters.
        Default: LOOSE_EPSILON (1e-10), which is appropriate for control parameters in [0,1].
        If None, uses NLopt default (0.0 = disabled).
    n_points_final_f : int, optional
        Target number of f control points in final iteration (only used in iterative mode).
        If specified, refinement_base is calculated as: ((n_points_final_f - 1) / (n_points_initial_f - 1))^(1/(n_iterations - 1))
        If None, uses refinement_base = 2.0 (default behavior: 2, 3, 5, 9, 17, ...)
        Example: n_points_final_f=17 with 5 iterations gives base ≈ 2.0
        Example: n_points_final_f=10 with 4 iterations gives base ≈ 2.08
    n_points_initial_f : int, optional
        Number of f control points in first iteration (only used in iterative mode).
        Default: 2
        Used with n_points_final_f to determine refinement_base.
        Example: n_points_initial_f=3, n_points_final_f=10, n_iterations=4 gives base ≈ 1.65
    initial_guess_s : list of float OR float, optional
        Enables dual optimization of both f and s when present.
        Direct mode (list): Initial s values at each control time.
            Must have same length as control_times.
            Each value must satisfy 0 ≤ s ≤ 1.
        Iterative refinement mode (float): Single initial s value for first iteration.
            Must satisfy 0 ≤ s ≤ 1.
    n_points_final_s : int, optional
        Target number of s control points in final iteration (iterative mode only).
        If None, uses same refinement base as f (derived from n_points_final_f and n_points_initial_f).
        Can differ from n_points_final_f to allow different temporal resolution for s.
    n_points_initial_s : int, optional
        Number of s control points in first iteration (only used in iterative mode).
        Default: 2
        Used with n_points_final_s to determine refinement_base for s.
    bounds_f : list of [float, float], optional
        Bounds for f values as [min, max]. Default: [0.0, 1.0]
        Example: [0.0, 0.8] to limit maximum abatement allocation to 80%
    bounds_s : list of [float, float], optional
        Bounds for s values as [min, max]. Default: [0.0, 1.0]
        Example: [0.2, 0.3] to constrain savings rate between 20% and 30%
    optimize_time_points : bool, optional
        Enable time adjustment optimization phase after standard iterations.
        When True, adjusts temporal spacing of control points while keeping values fixed.
        This allows optimizer to concentrate control points where changes matter most.
        Only applies to iterative refinement mode. Default: False (disabled)
    """
    max_evaluations: int
    control_times: object  # list or int
    initial_guess_f: object  # list or float
    algorithm: str = None
    ftol_rel: float = None
    ftol_abs: float = None
    xtol_rel: float = None
    xtol_abs: float = LOOSE_EPSILON
    n_points_final_f: int = None
    n_points_initial_f: int = 2
    initial_guess_s: object = None  # list or float, enables dual optimization if present
    n_points_final_s: int = None
    n_points_initial_s: int = 2
    bounds_f: list = None  # [min, max] for f, defaults to [0.0, 1.0]
    bounds_s: list = None  # [min, max] for s, defaults to [0.0, 1.0]
    optimize_time_points: bool = False

    def is_iterative_refinement(self):
        """
        Check if this configuration uses iterative refinement mode.

        Returns
        -------
        bool
            True if iterative refinement mode (control_times is int),
            False if direct mode (control_times is list)
        """
        return isinstance(self.control_times, int)

    def is_dual_optimization(self):
        """
        Check if this configuration optimizes both f and s.

        Returns
        -------
        bool
            True if optimizing both f and s (initial_guess_s is set),
            False if only optimizing f (s is fixed from time_functions)
        """
        return self.initial_guess_s is not None

    def is_direct_mode(self):
        """
        Check if this configuration uses direct multi-point mode.

        Returns
        -------
        bool
            True if direct mode (control_times is list),
            False if iterative refinement mode (control_times is int)
        """
        return isinstance(self.control_times, (list, np.ndarray))


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
        Integration control parameters
    optimization_params : OptimizationParameters
        Optimization control parameters
    initial_state : dict or None
        Initial conditions (computed automatically by integrate_model if None)
    control_function : callable
        Control function f(t) returning fraction allocated to abatement
    """
    run_name: str
    scalar_params: ScalarParameters
    time_functions: dict
    integration_params: IntegrationParameters
    optimization_params: OptimizationParameters
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
        'alpha', 'delta', 'psi1', 'psi2', 'y_damage_halfsat', 'k_climate',
        'eta', 'rho', 'Gini_initial', 'Gini_fract', 'Gini_restore', 'fract_gdp', 'theta2', 'mu_max',
        'A', 'L', 'sigma', 'theta1', 'f', 's'

    Notes
    -----
    The control function now returns a tuple (f, s) where both f and s can be
    optimized or prescribed. This enables dual optimization of abatement allocation
    and savings rate.
    """
    sp = config.scalar_params
    tf = config.time_functions

    params = {
        # Time
        't': t,

        # Scalar parameters
        'alpha': sp.alpha,
        'delta': sp.delta,
        'psi1': sp.psi1,
        'psi2': sp.psi2,
        'y_damage_halfsat': sp.y_damage_halfsat,
        'k_climate': sp.k_climate,
        'eta': sp.eta,
        'rho': sp.rho,
        'Gini_initial': sp.Gini_initial,
        'Gini_fract': sp.Gini_fract,
        'Gini_restore': sp.Gini_restore,
        'fract_gdp': sp.fract_gdp,
        'theta2': sp.theta2,
        'mu_max': sp.mu_max,

        # Time-dependent function evaluations
        'A': tf['A'](t),
        'L': tf['L'](t),
        'sigma': tf['sigma'](t),
        'theta1': tf['theta1'](t),
    }

    # Dual control function evaluation returns (f, s)
    f, s = config.control_function(t)
    params['f'] = f
    params['s'] = s

    return params


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
    elif func_type == 'double_exponential_growth':
        return create_double_exponential_growth(
            func_spec['initial_value'],
            func_spec['growth_rate_1'],
            func_spec['growth_rate_2'],
            func_spec['fract_1']
        )
    elif func_type == 'gompertz_growth':
        return create_gompertz_growth(
            func_spec['initial_value'],
            func_spec['final_value'],
            func_spec['adjustment_coefficient']
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


def _filter_description_keys(d):
    """
    Remove keys starting with '_' from dictionary (used for descriptions/comments).

    Parameters
    ----------
    d : dict
        Dictionary that may contain description keys

    Returns
    -------
    dict
        Dictionary with all keys starting with '_' removed
    """
    return {k: v for k, v in d.items() if not k.startswith('_')}


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
    - optimization_parameters: dict with max_evaluations
    - control_function: dict with type and parameters

    Initial state is computed automatically by integrate_model():
    - Ecum(0) = 0 (no cumulative emissions at start)
    - K(0) = (s·A(0)/δ)^(1/(1-α))·L(0) (steady-state capital)

    Keys starting with '_' are treated as comments/descriptions and ignored.

    See config_baseline.json for an example.
    """
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # Create scalar parameters (filter out description keys)
    scalar_params_data = _filter_description_keys(config_data['scalar_parameters'])
    scalar_params = ScalarParameters(**scalar_params_data)

    # Create time-dependent functions (filter out description keys)
    time_functions = {
        name: _create_time_function(_filter_description_keys(spec))
        for name, spec in config_data['time_functions'].items()
        if not name.startswith('_')
    }

    # Create integration parameters (filter out description keys)
    integration_params_data = _filter_description_keys(config_data['integration_parameters'])
    integration_params = IntegrationParameters(**integration_params_data)

    # Create optimization parameters (filter out description keys)
    optimization_params_data = _filter_description_keys(config_data['optimization_parameters'])
    optimization_params = OptimizationParameters(**optimization_params_data)

    # Create dual control function (f, s)
    # For optimization, these will be created from optimization results
    # For forward runs, these come from config
    if 'control_function' in config_data:
        control_function_data = _filter_description_keys(config_data['control_function'])
        f_control = _create_control_function(control_function_data)
    else:
        # Use initial_guess_f as default constant control function
        f_control = lambda t: optimization_params.initial_guess_f if isinstance(optimization_params.initial_guess_f, float) else optimization_params.initial_guess_f[0]

    if 's' in time_functions:
        s_time_function = time_functions['s']
    else:
        # Use initial_guess_s as default constant s function
        initial_s = optimization_params.initial_guess_s if optimization_params.initial_guess_s is not None else 0.24
        s_time_function = lambda t: initial_s if isinstance(initial_s, float) else initial_s[0]
        time_functions['s'] = s_time_function

    control_function = create_f_and_s_control_from_single(f_control, s_time_function)

    # Extract run name
    run_name = config_data['run_name']

    return ModelConfiguration(
        run_name=run_name,
        scalar_params=scalar_params,
        time_functions=time_functions,
        integration_params=integration_params,
        optimization_params=optimization_params,
        initial_state=None,
        control_function=control_function,
    )
