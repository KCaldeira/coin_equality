"""
Optimization framework for finding optimal allocation between redistribution and abatement.

This module provides control point parameterization for the control function f(t)
and optimization using NLopt to maximize discounted aggregate utility.
"""

import numpy as np
import nlopt
from scipy.interpolate import PchipInterpolator
from economic_model import integrate_model
from parameters import ModelConfiguration
from constants import EPSILON


def calculate_utility_weighted_times(n_points, config):
    """
    Calculate control point times weighted by contribution to discounted utility.

    Distributes control points to provide approximately equal contributions to
    the time-discounted aggregate utility integral. Concentrates points in early
    periods where discounting makes decisions more impactful.

    Parameters
    ----------
    n_points : int
        Number of control points to generate (must be >= 2)
    config : ModelConfiguration
        Model configuration containing time span, TFP function, and parameters

    Returns
    -------
    ndarray
        Control times from t_start to t_end, weighted by utility contribution

    Notes
    -----
    Algorithm:
    1. Compute average TFP growth rate: k_A = ln(A(t_end)/A(t_start)) / (t_end - t_start)
    2. Compute effective Consumption discount rate: r_c = ρ + η·k_A·(1-α)
    3. Generate times: t(k) = -(1/r_c)·ln(1 - (k/N)·(1 - exp(-r_c·t_end)))
       for k = 0, 1, ..., N where N = n_points - 1

    This ensures each interval contributes roughly equally to the discounted
    objective function, with more resolution where it matters most.
    """
    t_start = config.integration_params.t_start
    t_end = config.integration_params.t_end

    A_func = config.time_functions['A']
    A_start = A_func(t_start)
    A_end = A_func(t_end)

    rho = config.scalar_params.rho
    eta = config.scalar_params.eta
    alpha = config.scalar_params.alpha

    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) must be greater than t_start ({t_start})")

    if A_end <= 0 or A_start <= 0:
        raise ValueError(f"TFP must be positive: A(t_start)={A_start}, A(t_end)={A_end}")

    k_A = np.log(A_end / A_start) / (t_end - t_start)
    r_c = rho + eta * k_A * (1 - alpha)

    # use the mean of discount rate and pure rate of time preference
    # no theoretical justification
    r_c = (r_c + rho)/2.0

    N = n_points - 1
    k_values = np.arange(n_points)

    if abs(r_c) < EPSILON:
        times = t_start + (k_values / N) * (t_end - t_start)
    else:
        times = -(1.0 / r_c) * np.log(1.0 - (k_values / N) * (1.0 - np.exp(-r_c * t_end)))

    times[0] = t_start
    times[-1] = t_end

    return times


def interpolate_to_new_grid(old_times, old_values, new_times):
    """
    Interpolate control values to a new grid using PCHIP interpolation.

    Uses Piecewise Cubic Hermite Interpolating Polynomial for shape-preserving
    interpolation with continuous first derivatives. Clamps results to [0, 1]
    to ensure valid control values.

    Parameters
    ----------
    old_times : array_like
        Times at which old values are defined
    old_values : array_like
        Control values at old_times
    new_times : array_like
        Times at which to evaluate interpolated values

    Returns
    -------
    ndarray
        Interpolated values at new_times, clamped to [0, 1]

    Notes
    -----
    For new_times that match old_times, returns the exact old_values.
    For new_times beyond the range of old_times, uses constant extrapolation.
    Results are clamped to [0, 1] to ensure valid control function values.
    """
    old_times = np.asarray(old_times)
    old_values = np.asarray(old_values)
    new_times = np.asarray(new_times)

    if len(old_times) == 1:
        return np.full_like(new_times, old_values[0], dtype=float)

    interpolator = PchipInterpolator(old_times, old_values, extrapolate=False)
    new_values = np.where(
        new_times <= old_times[-1],
        interpolator(new_times),
        old_values[-1]
    )

    new_values = np.clip(new_values, 0.0, 1.0)

    return new_values


def evaluate_control_function(control_points, t):
    """
    Evaluate f(t) from control points using Pchip interpolation.

    Uses Pchip (Piecewise Cubic Hermite Interpolating Polynomial) for
    shape-preserving interpolation with continuous first derivatives.
    For t beyond the last control point, uses constant extrapolation.
    For a single control point, returns constant for all t.

    Parameters
    ----------
    control_points : list of tuples
        List of (time, value) tuples defining control points.
        Must have at least one point. Values should satisfy 0 <= value <= 1.
    t : float or array_like
        Time(s) at which to evaluate the control function

    Returns
    -------
    float or ndarray
        Control function value(s) at time(s) t, clamped to [0, 1]

    Notes
    -----
    Interpolation properties:
    - C¹ continuity (continuous first derivatives)
    - Shape-preserving and monotonicity-preserving
    - No overshoot beyond the range of control point values
    - Results are clamped to [0,1] to handle numerical precision issues

    Special cases:
    - Single control point [(t₀, f₀)]: returns f₀ for all t (constant)
    - For t > t_max: returns f(t_max) (constant extrapolation)

    Examples
    --------
    Single control point (constant trajectory):
    >>> f_func = lambda t: evaluate_control_function([(0, 0.5)], t)
    >>> f_func(100)  # Returns 0.5 for all t

    Multiple control points:
    >>> points = [(0, 0.2), (50, 0.8), (100, 0.6)]
    >>> f_func = lambda t: evaluate_control_function(points, t)
    >>> f_func(25)  # Interpolated value between 0.2 and 0.8
    """
    control_points = sorted(control_points)
    times = np.array([pt[0] for pt in control_points])
    values = np.array([pt[1] for pt in control_points])

    t_array = np.atleast_1d(t)

    if len(control_points) == 1:
        result = np.full_like(t_array, values[0], dtype=float)
    else:
        interpolator = PchipInterpolator(times, values, extrapolate=False)
        result = np.where(
            t_array <= times[-1],
            interpolator(t_array),
            values[-1]
        )

    result = np.clip(result, 0.0, 1.0)

    return result if np.ndim(t) > 0 else float(result[0])


def create_control_function_from_points(control_points):
    """
    Create a callable control function from control points.

    Parameters
    ----------
    control_points : list of tuples
        List of (time, value) tuples defining control points

    Returns
    -------
    callable
        Function f(t) that evaluates the control function at time t
    """
    return lambda t: evaluate_control_function(control_points, t)


def create_f_and_s_control_function_from_points(f_control_points, s_control_points):
    """
    Create a callable control function for both f and s from separate control points.

    Parameters
    ----------
    f_control_points : list of tuples
        List of (time, f_value) tuples defining f control points
    s_control_points : list of tuples
        List of (time, s_value) tuples defining s control points

    Returns
    -------
    callable
        Function returning (f(t), s(t)) tuple that evaluates both controls at time t

    Notes
    -----
    f and s are interpolated independently using their own control points.
    This allows different numbers of control points and different time spacing
    for each variable.
    """
    return lambda t: (
        evaluate_control_function(f_control_points, t),
        evaluate_control_function(s_control_points, t)
    )


class UtilityOptimizer:
    """
    Optimizer for finding optimal allocation between redistribution and abatement.

    Maximizes the discounted aggregate utility integral:
        max ∫₀^T e^(-ρt) · U(t) · L(t) dt

    where U(t) is mean utility and L(t) is population.

    The control function f(t) is parameterized by discrete control points,
    with interpolation and extrapolation handled by evaluate_control_function().
    """

    def __init__(self, base_config):
        """
        Initialize optimizer with base configuration.

        Parameters
        ----------
        base_config : ModelConfiguration
            Base model configuration. The control function will be replaced
            during optimization.
        """
        self.base_config = base_config
        self.n_evaluations = 0
        self.best_objective = -np.inf
        self.best_control_values = None
        self.degenerate_case = False
        self.degenerate_reason = None

    def calculate_objective(self, control_values, control_times, s_control_values=None, s_control_times=None):
        """
        Calculate the discounted aggregate utility for given control point values.

        Parameters
        ----------
        control_values : array_like
            Control function values (f) at control_times (one per control point)
        control_times : array_like
            Times at which f control points are placed
        s_control_values : array_like, optional
            Control function values (s) at s_control_times. If None, uses fixed s from time_functions.
        s_control_times : array_like, optional
            Times at which s control points are placed. If None, uses fixed s from time_functions.

        Returns
        -------
        float
            Discounted utility integral

        Notes
        -----
        Uses trapezoidal integration for the discounted utility integral.
        Control values are clamped to [0, 1] to handle numerical precision issues.

        If s_control_values and s_control_times are provided, creates dual control function
        with independent interpolation for f and s. Otherwise, uses fixed s(t) from configuration.
        """
        self.n_evaluations += 1

        control_values = np.clip(control_values, 0.0, 1.0)
        f_control_points = list(zip(control_times, control_values))

        if s_control_values is not None and s_control_times is not None:
            # f and s optimization mode: both f and s are control variables
            s_control_values = np.clip(s_control_values, 0.0, 1.0)
            s_control_points = list(zip(s_control_times, s_control_values))
            control_function = create_f_and_s_control_function_from_points(f_control_points, s_control_points)
        else:
            # Single optimization mode: only f is optimized, s is fixed from time_functions or initial_guess
            f_control = create_control_function_from_points(f_control_points)
            if 's' in self.base_config.time_functions:
                s_time_function = self.base_config.time_functions['s']
            else:
                # Use the s control function from the base config
                base_s_func = lambda t: self.base_config.control_function(t)[1]
                s_time_function = base_s_func
            from parameters import create_f_and_s_control_from_single
            control_function = create_f_and_s_control_from_single(f_control, s_time_function)

        config = ModelConfiguration(
            run_name=self.base_config.run_name,
            scalar_params=self.base_config.scalar_params,
            time_functions=self.base_config.time_functions,
            integration_params=self.base_config.integration_params,
            optimization_params=self.base_config.optimization_params,
            initial_state=self.base_config.initial_state,
            control_function=control_function
        )

        # Use store_detailed_output=False during optimization for better performance
        results = integrate_model(config, store_detailed_output=False)

        rho = self.base_config.scalar_params.rho
        t = results['t']
        U = results['U']
        L = results['L']

        discount_factors = np.exp(-rho * t)
        integrand = discount_factors * U * L

        # np.trapezoid for numerically integrates by drawing a straight line between points
        objective_value = np.trapezoid(integrand, t)

        if objective_value > self.best_objective:
            self.best_objective = objective_value
            self.best_control_values = control_values.copy()

        return objective_value

    def sensitivity_analysis(self, f_values):
        """
        Evaluate objective function at multiple fixed f values.

        Useful for understanding the objective function landscape and
        validating optimization results.

        Parameters
        ----------
        f_values : array_like
            Array of f values to evaluate (each should be in [0, 1])

        Returns
        -------
        dict
            Results containing:
            - 'f_values': input f values
            - 'objectives': corresponding objective values
            - 'n_evaluations': total evaluations performed
        """
        self.n_evaluations = 0
        control_times = [self.base_config.integration_params.t_start]

        objectives = []
        for f_val in f_values:
            obj = self.calculate_objective([f_val], control_times)
            objectives.append(obj)

        return {
            'f_values': np.array(f_values),
            'objectives': np.array(objectives),
            'n_evaluations': self.n_evaluations
        }

    def optimize_control_points(self, control_times, initial_guess, max_evaluations,
                                         algorithm=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, xtol_abs=None):
        """
        Optimize allocation with multiple control points (time-varying trajectory).

        Finds optimal control point values that maximize discounted utility.

        Parameters
        ----------
        control_times : array_like
            Times at which control points are placed
        initial_guess : array_like
            Initial guess for control values at each control time
        max_evaluations : int
            Maximum number of objective function evaluations
        algorithm : str, optional
            NLopt algorithm name (e.g., 'LN_SBPLX', 'LN_BOBYQA', 'GN_ISRES').
            If None, defaults to 'LN_SBPLX'.
        ftol_rel : float, optional
            Relative tolerance on objective function changes (None = use NLopt default)
        ftol_abs : float, optional
            Absolute tolerance on objective function changes (None = use NLopt default)
        xtol_rel : float, optional
            Relative tolerance on parameter changes (None = use NLopt default)
        xtol_abs : float, optional
            Absolute tolerance on parameter changes (None = use NLopt default)

        Returns
        -------
        dict
            Optimization results containing:
            - 'optimal_values': optimal control values at each control time
            - 'optimal_objective': maximum utility achieved
            - 'n_evaluations': number of objective evaluations used
            - 'control_points': list of (time, value) tuples
            - 'status': optimization status string
            - 'termination_code': NLopt termination code
            - 'termination_name': human-readable termination reason
            - 'algorithm': algorithm name used
        """
        self.n_evaluations = 0
        self.best_objective = -np.inf
        self.best_control_values = None
        self.degenerate_case = False
        self.degenerate_reason = None

        if algorithm is None:
            algorithm = 'LN_SBPLX'

        fract_gdp = self.base_config.scalar_params.fract_gdp
        if abs(fract_gdp) < EPSILON:
            self.degenerate_case = True
            self.degenerate_reason = "fract_gdp = 0: No income available for redistribution or abatement. Control values have no effect on outcome."
            control_times_array = np.array(control_times)
            initial_guess_array = np.array(initial_guess)
            obj = self.calculate_objective(initial_guess_array, control_times_array)
            control_points = list(zip(control_times_array, initial_guess_array))
            return {
                'optimal_values': initial_guess_array,
                'optimal_objective': obj,
                'n_evaluations': self.n_evaluations,
                'control_points': control_points,
                'status': 'degenerate',
                'termination_code': None,
                'termination_name': 'DEGENERATE_CASE',
                'algorithm': algorithm
            }

        n_points = len(control_times)
        control_times = np.array(control_times)

        def objective_wrapper(x, grad):
            return self.calculate_objective(x, control_times)

        # Get bounds from config, default to [0.0, 1.0]
        bounds_f = self.base_config.optimization_params.bounds_f if self.base_config.optimization_params.bounds_f is not None else [0.0, 1.0]

        nlopt_algorithm = getattr(nlopt, algorithm)
        opt = nlopt.opt(nlopt_algorithm, n_points)
        opt.set_lower_bounds(np.full(n_points, bounds_f[0]))
        opt.set_upper_bounds(np.full(n_points, bounds_f[1]))
        opt.set_max_objective(objective_wrapper)
        opt.set_maxeval(max_evaluations)

        if ftol_rel is not None:
            opt.set_ftol_rel(ftol_rel)
        if ftol_abs is not None:
            opt.set_ftol_abs(ftol_abs)
        if xtol_rel is not None:
            opt.set_xtol_rel(xtol_rel)
        if xtol_abs is not None:
            opt.set_xtol_abs(xtol_abs)

        x0 = np.array(initial_guess)
        optimal_x = opt.optimize(x0)
        optimal_f = opt.last_optimum_value()
        termination_code = opt.last_optimize_result()

        termination_names = {
            1: 'SUCCESS',
            2: 'STOPVAL_REACHED',
            3: 'FTOL_REACHED',
            4: 'XTOL_REACHED',
            5: 'MAXEVAL_REACHED',
            6: 'MAXTIME_REACHED',
            -1: 'FAILURE',
            -2: 'INVALID_ARGS',
            -3: 'OUT_OF_MEMORY',
            -4: 'ROUNDOFF_LIMITED',
            -5: 'FORCED_STOP'
        }
        termination_name = termination_names.get(termination_code, f'UNKNOWN_{termination_code}')

        control_points = list(zip(control_times, optimal_x))

        return {
            'optimal_values': optimal_x,
            'optimal_objective': optimal_f,
            'n_evaluations': self.n_evaluations,
            'control_points': control_points,
            'status': 'success',
            'termination_code': termination_code,
            'termination_name': termination_name,
            'algorithm': algorithm
        }

    def optimize_control_points_f_and_s(self, f_control_times, f_initial_guess,
                                     s_control_times, s_initial_guess,
                                     max_evaluations,
                                     algorithm=None, ftol_rel=None, ftol_abs=None,
                                     xtol_rel=None, xtol_abs=None):
        """
        Optimize both f and s control points simultaneously.

        Parameters
        ----------
        f_control_times : array_like
            Times at which f control points are placed
        f_initial_guess : array_like
            Initial guess for f values at each f control time
        s_control_times : array_like
            Times at which s control points are placed
        s_initial_guess : array_like
            Initial guess for s values at each s control time
        max_evaluations : int
            Maximum number of objective function evaluations
        algorithm : str, optional
            NLopt algorithm name
        ftol_rel, ftol_abs, xtol_rel, xtol_abs : float, optional
            Tolerance parameters

        Returns
        -------
        dict
            Optimization results with separate f and s control points
        """
        self.n_evaluations = 0
        self.best_objective = -np.inf
        self.best_control_values = None
        self.degenerate_case = False
        self.degenerate_reason = None

        if algorithm is None:
            algorithm = 'LN_SBPLX'

        fract_gdp = self.base_config.scalar_params.fract_gdp
        if abs(fract_gdp) < EPSILON:
            self.degenerate_case = True
            self.degenerate_reason = "fract_gdp = 0: No income available for redistribution or abatement."
            f_control_times_array = np.array(f_control_times)
            f_initial_guess_array = np.array(f_initial_guess)
            s_control_times_array = np.array(s_control_times)
            s_initial_guess_array = np.array(s_initial_guess)
            obj = self.calculate_objective(
                f_initial_guess_array, f_control_times_array,
                s_initial_guess_array, s_control_times_array
            )
            f_control_points = list(zip(f_control_times_array, f_initial_guess_array))
            s_control_points = list(zip(s_control_times_array, s_initial_guess_array))
            return {
                'optimal_values': f_initial_guess_array,
                's_optimal_values': s_initial_guess_array,
                'optimal_objective': obj,
                'n_evaluations': self.n_evaluations,
                'control_points': f_control_points,
                's_control_points': s_control_points,
                'status': 'degenerate',
                'termination_code': None,
                'termination_name': 'DEGENERATE_CASE',
                'algorithm': algorithm
            }

        n_f_points = len(f_control_times)
        n_s_points = len(s_control_times)
        n_total_points = n_f_points + n_s_points

        f_control_times = np.array(f_control_times)
        s_control_times = np.array(s_control_times)

        def objective_wrapper(x, grad):
            # Split combined vector into f and s components
            f_values = x[:n_f_points]
            s_values = x[n_f_points:]
            return self.calculate_objective(f_values, f_control_times,
                                           s_values, s_control_times)

        # Get bounds from config, default to [0.0, 1.0]
        bounds_f = self.base_config.optimization_params.bounds_f if self.base_config.optimization_params.bounds_f is not None else [0.0, 1.0]
        bounds_s = self.base_config.optimization_params.bounds_s if self.base_config.optimization_params.bounds_s is not None else [0.0, 1.0]

        # Combine bounds: [f_min, f_min, ..., s_min, s_min, ...]
        lower_bounds = np.concatenate([np.full(n_f_points, bounds_f[0]), np.full(n_s_points, bounds_s[0])])
        upper_bounds = np.concatenate([np.full(n_f_points, bounds_f[1]), np.full(n_s_points, bounds_s[1])])

        nlopt_algorithm = getattr(nlopt, algorithm)
        opt = nlopt.opt(nlopt_algorithm, n_total_points)
        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)
        opt.set_max_objective(objective_wrapper)
        opt.set_maxeval(max_evaluations)

        if ftol_rel is not None:
            opt.set_ftol_rel(ftol_rel)
        if ftol_abs is not None:
            opt.set_ftol_abs(ftol_abs)
        if xtol_rel is not None:
            opt.set_xtol_rel(xtol_rel)
        if xtol_abs is not None:
            opt.set_xtol_abs(xtol_abs)

        # Combine f and s initial guesses
        x0 = np.concatenate([np.array(f_initial_guess), np.array(s_initial_guess)])

        # Ensure x0 is within bounds (clip to bounds to handle floating point precision issues)
        x0 = np.clip(x0, lower_bounds, upper_bounds)

        optimal_x = opt.optimize(x0)
        optimal_f_val = opt.last_optimum_value()
        termination_code = opt.last_optimize_result()

        termination_names = {
            1: 'SUCCESS',
            2: 'STOPVAL_REACHED',
            3: 'FTOL_REACHED',
            4: 'XTOL_REACHED',
            5: 'MAXEVAL_REACHED',
            6: 'MAXTIME_REACHED',
            -1: 'FAILURE',
            -2: 'INVALID_ARGS',
            -3: 'OUT_OF_MEMORY',
            -4: 'ROUNDOFF_LIMITED',
            -5: 'FORCED_STOP'
        }
        termination_name = termination_names.get(termination_code, f'UNKNOWN_{termination_code}')

        # Split optimal values back into f and s
        optimal_f_values = optimal_x[:n_f_points]
        optimal_s_values = optimal_x[n_f_points:]

        f_control_points = list(zip(f_control_times, optimal_f_values))
        s_control_points = list(zip(s_control_times, optimal_s_values))

        return {
            'optimal_values': optimal_f_values,
            's_optimal_values': optimal_s_values,
            'optimal_objective': optimal_f_val,
            'n_evaluations': self.n_evaluations,
            'control_points': f_control_points,
            's_control_points': s_control_points,
            'status': 'success',
            'termination_code': termination_code,
            'termination_name': termination_name,
            'algorithm': algorithm
        }

    def optimize_with_iterative_refinement(self, n_iterations, initial_guess_scalar,
                                          max_evaluations, algorithm=None,
                                          ftol_rel=None, ftol_abs=None,
                                          xtol_rel=None, xtol_abs=None,
                                          n_points_final=None,
                                          initial_guess_s_scalar=None,
                                          n_points_final_s=None):
        """
        Optimize using iterative refinement with progressively finer control grids.

        Performs a sequence of optimizations with increasing numbers of control points.
        Each iteration uses PCHIP interpolation of the previous solution to initialize
        the optimization, providing better convergence than cold-starting with many
        control points.

        Parameters
        ----------
        n_iterations : int
            Number of refinement iterations to perform.
            Iteration k produces round(1 + base^(k-1)) control points.
        initial_guess_scalar : float
            Initial f value for all control points in first iteration.
            Must satisfy 0 ≤ f ≤ 1.
        max_evaluations : int
            Maximum objective function evaluations per iteration
        algorithm : str, optional
            NLopt algorithm name. If None, defaults to 'LN_SBPLX'.
        ftol_rel : float, optional
            Relative tolerance on objective function changes
        ftol_abs : float, optional
            Absolute tolerance on objective function changes
        xtol_rel : float, optional
            Relative tolerance on parameter changes
        xtol_abs : float, optional
            Absolute tolerance on parameter changes
        n_points_final : int, optional
            Target number of control points in final iteration.
            If specified, base = (n_points_final - 1)^(1/(n_iterations - 1))
            If None, uses base = 2.0 (default: 2, 3, 5, 9, 17, ...)

        Returns
        -------
        dict
            Optimization results containing:
            - 'optimal_values': optimal control values from final iteration
            - 'optimal_objective': maximum utility achieved
            - 'n_evaluations': total evaluations across all iterations
            - 'control_points': list of (time, value) tuples from final iteration
            - 'status': optimization status string
            - 'algorithm': algorithm name used
            - 'n_iterations': number of iterations performed
            - 'iteration_history': list of results from each iteration
            - 'iteration_control_grids': control times used at each iteration
            - 'refinement_base': base used for point growth

        Notes
        -----
        Iteration schedule (default base=2.0):
        - Iteration 1: 2 control points at [t_start, t_end]
        - Iteration 2: 3 control points
        - Iteration k: round(1 + base^(k-1)) control points

        Initial guess strategy:
        - First iteration: uses initial_guess_scalar for all points
        - Subsequent iterations: uses previous optimal values at existing points,
          PCHIP interpolation for new midpoints
        """
        # Determine if f and s optimization is enabled
        optimize_f_and_s = initial_guess_s_scalar is not None

        # Calculate refinement base for f
        if n_points_final is not None:
            if n_iterations <= 1:
                refinement_base_f = 2.0
            else:
                refinement_base_f = (n_points_final - 1) ** (1.0 / (n_iterations - 1))
        else:
            refinement_base_f = 2.0

        # Calculate refinement base for s (if optimizing both f and s)
        if optimize_f_and_s:
            if n_points_final_s is not None:
                if n_iterations <= 1:
                    refinement_base_s = 2.0
                else:
                    refinement_base_s = (n_points_final_s - 1) ** (1.0 / (n_iterations - 1))
            else:
                refinement_base_s = refinement_base_f  # Use same base as f if not specified

        print(f"\nIterative refinement: {n_iterations} iterations, base_f = {refinement_base_f:.4f}")
        if n_points_final is not None:
            print(f"Target final f points: {n_points_final}")
        if optimize_f_and_s:
            print(f"Optimizing both f and s: base_s = {refinement_base_s:.4f}")
            if n_points_final_s is not None:
                print(f"Target final s points: {n_points_final_s}")

        iteration_history = []
        iteration_f_control_grids = []
        iteration_s_control_grids = [] if optimize_f_and_s else None
        total_evaluations = 0

        for iteration in range(1, n_iterations + 1):
            # Calculate f control points
            n_points_f = round(1 + refinement_base_f**(iteration - 1))
            f_control_times = calculate_utility_weighted_times(n_points_f, self.base_config)

            if iteration == 1:
                f_initial_guess = np.full(n_points_f, initial_guess_scalar)
            else:
                old_f_times = iteration_f_control_grids[-1]
                old_f_values = iteration_history[-1]['optimal_values']
                f_initial_guess = interpolate_to_new_grid(old_f_times, old_f_values, f_control_times)

            iteration_f_control_grids.append(f_control_times.copy())

            # Calculate s control points (if optimizing both f and s)
            if optimize_f_and_s:
                n_points_s = round(1 + refinement_base_s**(iteration - 1))
                s_control_times = calculate_utility_weighted_times(n_points_s, self.base_config)

                if iteration == 1:
                    s_initial_guess = np.full(n_points_s, initial_guess_s_scalar)
                else:
                    old_s_times = iteration_s_control_grids[-1]
                    old_s_values = iteration_history[-1]['s_optimal_values']
                    s_initial_guess = interpolate_to_new_grid(old_s_times, old_s_values, s_control_times)

                iteration_s_control_grids.append(s_control_times.copy())

            # Print iteration info
            print(f"\n{'=' * 80}")
            print(f"  ITERATION {iteration}/{n_iterations}")
            print(f"\n  f (abatement fraction) - OPTIMIZED:")
            print(f"    Control points: {n_points_f}")
            print(f"    Time points: {f_control_times}")
            print(f"    Initial values: {f_initial_guess}")

            if optimize_f_and_s:
                print(f"\n  s (savings rate) - OPTIMIZED:")
                print(f"    Control points: {n_points_s}")
                print(f"    Time points: {s_control_times}")
                print(f"    Initial values: {s_initial_guess}")
            print(f"{'=' * 80}\n")

            # Run optimization
            if optimize_f_and_s:
                opt_result = self.optimize_control_points_f_and_s(
                    f_control_times,
                    f_initial_guess,
                    s_control_times,
                    s_initial_guess,
                    max_evaluations,
                    algorithm=algorithm,
                    ftol_rel=ftol_rel,
                    ftol_abs=ftol_abs,
                    xtol_rel=xtol_rel,
                    xtol_abs=xtol_abs
                )
            else:
                opt_result = self.optimize_control_points(
                    f_control_times,
                    f_initial_guess,
                    max_evaluations,
                    algorithm=algorithm,
                    ftol_rel=ftol_rel,
                    ftol_abs=ftol_abs,
                    xtol_rel=xtol_rel,
                    xtol_abs=xtol_abs
                )

            opt_result['iteration'] = iteration
            opt_result['n_control_points'] = n_points_f
            if optimize_f_and_s:
                opt_result['n_s_control_points'] = n_points_s
            iteration_history.append(opt_result)
            total_evaluations += opt_result['n_evaluations']

            print(f"\nIteration {iteration} complete:")
            print(f"  Objective: {opt_result['optimal_objective']:.6e}")
            print(f"  Evaluations: {opt_result['n_evaluations']}")
            print(f"  Status: {opt_result['termination_name']}")

        final_result = iteration_history[-1]

        result = {
            'optimal_values': final_result['optimal_values'],
            'optimal_objective': final_result['optimal_objective'],
            'n_evaluations': total_evaluations,
            'control_points': final_result['control_points'],
            'status': 'success',
            'algorithm': algorithm if algorithm is not None else 'LN_SBPLX',
            'n_iterations': n_iterations,
            'iteration_history': iteration_history,
            'iteration_control_grids': iteration_f_control_grids,
            'refinement_base': refinement_base_f
        }

        if optimize_f_and_s:
            result['s_optimal_values'] = final_result['s_optimal_values']
            result['s_control_points'] = final_result['s_control_points']
            result['iteration_s_control_grids'] = iteration_s_control_grids
            result['refinement_base_s'] = refinement_base_s

        return result
