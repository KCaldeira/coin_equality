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
        Control function value(s) at time(s) t

    Notes
    -----
    Interpolation properties:
    - C¹ continuity (continuous first derivatives)
    - Shape-preserving and monotonicity-preserving
    - No overshoot beyond the range of control point values
    - Ensures f(t) ∈ [0,1] when all control points satisfy 0 ≤ fᵢ ≤ 1

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

    def calculate_objective(self, control_values, control_times):
        """
        Calculate the discounted aggregate utility for given control point values.

        Parameters
        ----------
        control_values : array_like
            Control function values at control_times (one per control point)
        control_times : array_like
            Times at which control points are placed

        Returns
        -------
        float
            Discounted utility integral

        Notes
        -----
        Uses trapezoidal integration for the discounted utility integral.
        """
        self.n_evaluations += 1

        control_points = list(zip(control_times, control_values))
        control_function = create_control_function_from_points(control_points)

        config = ModelConfiguration(
            run_name=self.base_config.run_name,
            scalar_params=self.base_config.scalar_params,
            time_functions=self.base_config.time_functions,
            integration_params=self.base_config.integration_params,
            optimization_params=self.base_config.optimization_params,
            initial_state=self.base_config.initial_state,
            control_function=control_function
        )

        results = integrate_model(config)

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

    def optimize_single_control_point(self, initial_guess, max_evaluations):
        """
        Optimize allocation with a single control point (constant trajectory).

        Finds optimal constant f₀ ∈ [0, 1] that maximizes discounted utility.

        Parameters
        ----------
        initial_guess : float
            Initial guess for f₀
        max_evaluations : int
            Maximum number of objective function evaluations

        Returns
        -------
        dict
            Optimization results containing:
            - 'optimal_value': optimal control value f₀
            - 'optimal_objective': maximum utility achieved
            - 'n_evaluations': number of objective evaluations used
            - 'control_points': list containing [(0, f₀)]
            - 'status': optimization status message
        """
        self.n_evaluations = 0
        self.best_objective = -np.inf
        self.best_control_values = None

        control_times = [self.base_config.integration_params.t_start]

        def objective_wrapper(x, grad):
            return self.calculate_objective(x, control_times)

        opt = nlopt.opt(nlopt.LN_BOBYQA, 1)
        opt.set_lower_bounds([0.0])
        opt.set_upper_bounds([1.0])
        opt.set_max_objective(objective_wrapper)
        opt.set_maxeval(max_evaluations)
        opt.set_xtol_rel(1e-6)

        x0 = np.array([initial_guess])
        optimal_x = opt.optimize(x0)
        optimal_f = opt.last_optimum_value()

        return {
            'optimal_value': float(optimal_x[0]),
            'optimal_objective': optimal_f,
            'n_evaluations': self.n_evaluations,
            'control_points': [(control_times[0], optimal_x[0])],
            'status': 'success'
        }

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

    def optimize_multiple_control_points(self, control_times, initial_guess, max_evaluations):
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

        Returns
        -------
        dict
            Optimization results containing:
            - 'optimal_values': optimal control values at each control time
            - 'optimal_objective': maximum utility achieved
            - 'n_evaluations': number of objective evaluations used
            - 'control_points': list of (time, value) tuples
            - 'status': optimization status message
        """
        self.n_evaluations = 0
        self.best_objective = -np.inf
        self.best_control_values = None

        n_points = len(control_times)
        control_times = np.array(control_times)

        def objective_wrapper(x, grad):
            return self.calculate_objective(x, control_times)

        opt = nlopt.opt(nlopt.LN_BOBYQA, n_points)
        opt.set_lower_bounds(np.zeros(n_points))
        opt.set_upper_bounds(np.ones(n_points))
        opt.set_max_objective(objective_wrapper)
        opt.set_maxeval(max_evaluations)
        opt.set_xtol_rel(1e-6)

        x0 = np.array(initial_guess)
        optimal_x = opt.optimize(x0)
        optimal_f = opt.last_optimum_value()

        control_points = list(zip(control_times, optimal_x))

        return {
            'optimal_values': optimal_x,
            'optimal_objective': optimal_f,
            'n_evaluations': self.n_evaluations,
            'control_points': control_points,
            'status': 'success'
        }
