"""
Basis function representation for control variables (f and s).

Instead of interpolating through discrete control points, represent control
variables as linear combinations of basis functions (e.g., Chebyshev polynomials).
This reduces the dimensionality of the optimization problem and enforces smoothness.

Key advantages:
- Fewer decision variables (10-20 coefficients vs 50-100 control points)
- Smoother solutions by construction
- Better optimization convergence in lower-dimensional space
- Automatic bound satisfaction with sigmoid transformation
"""

import numpy as np
from numpy.polynomial import chebyshev, legendre


def sigmoid(x):
    """
    Numerically stable sigmoid function.

    Returns σ(x) = 1 / (1 + exp(-x))
    """
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def inverse_sigmoid(y, eps=1e-15):
    """
    Inverse sigmoid function (logit).

    Returns x such that sigmoid(x) = y
    x = log(y / (1 - y))

    Parameters
    ----------
    y : float or array
        Value in (0, 1)
    eps : float
        Small value to clip y away from 0 and 1 to avoid infinities

    Returns
    -------
    float or array
        Inverse sigmoid value
    """
    y_clipped = np.clip(y, eps, 1 - eps)
    return np.log(y_clipped / (1 - y_clipped))


def physical_to_normalized(value, value_min, value_max):
    """
    Convert physical value to normalized value in [0, 1].

    normalized = (value - value_min) / (value_max - value_min)

    Parameters
    ----------
    value : float or array
        Physical value in [value_min, value_max]
    value_min : float
        Minimum physical value
    value_max : float
        Maximum physical value

    Returns
    -------
    float or array
        Normalized value in [0, 1]
    """
    return (value - value_min) / (value_max - value_min)


def normalized_to_physical(normalized, value_min, value_max):
    """
    Convert normalized value in [0, 1] to physical value.

    value = value_min + (value_max - value_min) * normalized

    Parameters
    ----------
    normalized : float or array
        Normalized value in [0, 1]
    value_min : float
        Minimum physical value
    value_max : float
        Maximum physical value

    Returns
    -------
    float or array
        Physical value in [value_min, value_max]
    """
    return value_min + (value_max - value_min) * normalized


def physical_to_sigmoid_space(value, value_min, value_max, eps=1e-15):
    """
    Convert physical value to sigmoid space (pre-sigmoid value g).

    value = value_min + (value_max - value_min) * sigmoid(g)

    Solving for g:
    g = inverse_sigmoid((value - value_min) / (value_max - value_min))

    Parameters
    ----------
    value : float or array
        Physical value in [value_min, value_max]
    value_min : float
        Minimum physical value
    value_max : float
        Maximum physical value
    eps : float
        Small value to avoid infinities at boundaries

    Returns
    -------
    float or array
        Sigmoid-space value g
    """
    normalized = physical_to_normalized(value, value_min, value_max)
    return inverse_sigmoid(normalized, eps)


def sigmoid_space_to_physical(g, value_min, value_max):
    """
    Convert sigmoid-space value g to physical value.

    value = value_min + (value_max - value_min) * sigmoid(g)

    Parameters
    ----------
    g : float or array
        Sigmoid-space value (can be any real number)
    value_min : float
        Minimum physical value
    value_max : float
        Maximum physical value

    Returns
    -------
    float or array
        Physical value in [value_min, value_max]
    """
    return normalized_to_physical(sigmoid(g), value_min, value_max)


def calculate_coefficient_bounds(eps):
    """
    Calculate bounds for Chebyshev coefficients to keep sigmoid-transformed
    values away from 0 and 1.

    We want sigmoid(g) to stay in [eps, 1-eps], which means:
    g should stay in [log(eps/(1-eps)), log((1-eps)/eps)]

    Parameters
    ----------
    eps : float
        Small value (e.g., 1e-10) to keep sigmoid away from boundaries

    Returns
    -------
    tuple of (lower_bound, upper_bound)
        Symmetric bounds for coefficients
    """
    lower = np.log(eps / (1 - eps))
    upper = -lower  # Symmetric: log((1-eps)/eps) = -log(eps/(1-eps))
    return lower, upper


class BasisControlFunction:
    """
    Represents a control variable (e.g., f or s) using basis functions.

    value(t) = value_min + (value_max - value_min) * sigmoid(Σ c_i * φ_i(τ))

    where τ = 2*(t - t_start)/(t_end - t_start) - 1  ∈ [-1, 1]

    The sigmoid transformation ensures bounds are satisfied for all t.
    """

    def __init__(self, t_start, t_end, n_basis, value_min, value_max,
                 basis_type='chebyshev', eps=1e-10):
        """
        Parameters
        ----------
        t_start : float
            Start time of simulation
        t_end : float
            End time of simulation
        n_basis : int
            Number of basis functions (decision variables)
        value_min : float
            Minimum allowed value for this control variable
        value_max : float
            Maximum allowed value for this control variable
        basis_type : str
            Type of basis functions: 'chebyshev', 'legendre', or 'power'
        eps : float
            Small value to keep sigmoid-transformed values away from boundaries
        """
        self.t_start = t_start
        self.t_end = t_end
        self.n_basis = n_basis
        self.value_min = value_min
        self.value_max = value_max
        self.basis_type = basis_type
        self.eps = eps

        # Calculate coefficient bounds
        self.coeff_lower, self.coeff_upper = calculate_coefficient_bounds(eps)

    def time_to_normalized(self, t):
        """
        Convert physical time t ∈ [t_start, t_end] to normalized τ ∈ [-1, 1].
        """
        return 2.0 * (t - self.t_start) / (self.t_end - self.t_start) - 1.0

    def evaluate_basis(self, tau):
        """
        Evaluate all basis functions at normalized coordinate τ.

        Returns
        -------
        array of shape (n_basis,)
            Values of each basis function at τ
        """
        if self.basis_type == 'chebyshev':
            # Chebyshev polynomials T_i(τ)
            # T_0(τ) = 1, T_1(τ) = τ, T_2(τ) = 2τ² - 1, ...
            basis_values = np.array([chebyshev.chebval(tau, [0]*i + [1])
                                     for i in range(self.n_basis)])

        elif self.basis_type == 'legendre':
            # Legendre polynomials P_i(τ)
            basis_values = np.array([legendre.legval(tau, [0]*i + [1])
                                     for i in range(self.n_basis)])

        elif self.basis_type == 'power':
            # Simple power series: 1, τ, τ², τ³, ...
            basis_values = np.array([tau**i for i in range(self.n_basis)])

        else:
            raise ValueError(f"Unknown basis type: {self.basis_type}")

        return basis_values

    def evaluate(self, t, coefficients):
        """
        Evaluate control function at time t with given coefficients.

        value(t) = value_min + (value_max - value_min) * sigmoid(Σ c_i * T_i(τ))

        Parameters
        ----------
        t : float or array
            Time value(s) at which to evaluate
        coefficients : array of shape (n_basis,)
            Basis function coefficients

        Returns
        -------
        float or array
            Control variable value(s) at time t
        """
        # Normalize time
        tau = self.time_to_normalized(t)

        # Evaluate basis functions and compute g = Σ c_i * φ_i(τ)
        is_scalar = np.isscalar(t)
        if is_scalar:
            basis_values = self.evaluate_basis(tau)
            g = np.dot(coefficients, basis_values)
        else:
            # Vectorized for multiple time points
            tau_array = np.atleast_1d(tau)
            g = np.zeros_like(tau_array)
            for i, tau_i in enumerate(tau_array):
                basis_values = self.evaluate_basis(tau_i)
                g[i] = np.dot(coefficients, basis_values)

        # Apply sigmoid transformation
        value = sigmoid_space_to_physical(g, self.value_min, self.value_max)

        return float(value) if is_scalar else value

    def create_initial_guess(self, initial_value):
        """
        Create initial guess for coefficients to represent constant function.

        For value(t) = initial_value (constant), we need:
        - sigmoid(c_0 * T_0(τ)) = (initial_value - value_min) / (value_max - value_min)
        - Since T_0(τ) = 1, we need: sigmoid(c_0) = target
        - Therefore: c_0 = inverse_sigmoid(target)
        - All other c_i = 0

        Parameters
        ----------
        initial_value : float
            Constant value for this control variable

        Returns
        -------
        array of shape (n_basis,)
            Initial coefficient values
        """
        # Convert to normalized space
        target = physical_to_normalized(initial_value, self.value_min, self.value_max)

        # Compute c_0 using inverse sigmoid
        c_0 = inverse_sigmoid(target, self.eps)

        # Create coefficient array: [c_0, 0, 0, ..., 0]
        coefficients = np.zeros(self.n_basis)
        coefficients[0] = c_0

        return coefficients

    def get_coefficient_bounds(self):
        """
        Get optimization bounds for coefficients.

        Returns
        -------
        tuple of (lower_bounds, upper_bounds)
            Arrays of bounds for each coefficient
        """
        lower_bounds = np.full(self.n_basis, self.coeff_lower)
        upper_bounds = np.full(self.n_basis, self.coeff_upper)
        return lower_bounds, upper_bounds


def create_dual_basis_control(t_start, t_end, n_basis_f, n_basis_s,
                               f_min, f_max,
                               s_min, s_max,
                               basis_type='chebyshev',
                               initial_f=None, initial_s=None,
                               eps=1e-10):
    """
    Create basis function controls for both f (abatement fraction) and s (savings rate).

    Parameters
    ----------
    t_start : float
        Start time
    t_end : float
        End time
    n_basis_f : int
        Number of basis functions for f (abatement fraction)
    n_basis_s : int
        Number of basis functions for s (savings rate)
    f_min, f_max : float
        Bounds for f (abatement fraction)
    s_min, s_max : float
        Bounds for s (savings rate)
    basis_type : str
        Type of basis functions
    initial_f : float, optional
        Initial constant value for f (if provided, creates initial guess)
    initial_s : float, optional
        Initial constant value for s (if provided, creates initial guess)
    eps : float
        Epsilon for coefficient bounds

    Returns
    -------
    tuple of (f_control, s_control, initial_coefficients)
        f_control : BasisControlFunction for abatement fraction
        s_control : BasisControlFunction for savings rate
        initial_coefficients : array (if initial_f and initial_s provided)
            Concatenated initial guess for all coefficients
    """
    f_control = BasisControlFunction(
        t_start, t_end, n_basis_f,
        value_min=f_min, value_max=f_max,
        basis_type=basis_type, eps=eps
    )

    s_control = BasisControlFunction(
        t_start, t_end, n_basis_s,
        value_min=s_min, value_max=s_max,
        basis_type=basis_type, eps=eps
    )

    # Create initial guesses if values provided
    if initial_f is not None and initial_s is not None:
        f_coeffs_init = f_control.create_initial_guess(initial_f)
        s_coeffs_init = s_control.create_initial_guess(initial_s)
        initial_coefficients = np.concatenate([f_coeffs_init, s_coeffs_init])
        return f_control, s_control, initial_coefficients
    else:
        return f_control, s_control


def basis_control_function_factory(f_control, s_control):
    """
    Create a control function compatible with existing optimization infrastructure.

    Parameters
    ----------
    f_control : BasisControlFunction
    s_control : BasisControlFunction

    Returns
    -------
    callable
        Function with signature (f, s) = control_func(t, coefficients)
        where coefficients = [f_coeffs, s_coeffs] concatenated
    """
    n_f = f_control.n_basis

    def control_func(t, coefficients):
        """Evaluate both f and s at time t using Chebyshev basis functions."""
        f_coeffs = coefficients[:n_f]
        s_coeffs = coefficients[n_f:]

        f = f_control.evaluate(t, f_coeffs)
        s = s_control.evaluate(t, s_coeffs)

        return f, s

    return control_func
