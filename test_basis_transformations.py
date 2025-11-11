"""
Test basis function transformations and coefficient bounds.
"""

import numpy as np
from constants import EPSILON, LOOSE_EPSILON
from basis_control import (
    sigmoid, inverse_sigmoid,
    physical_to_normalized, normalized_to_physical,
    physical_to_sigmoid_space, sigmoid_space_to_physical,
    calculate_coefficient_bounds,
    BasisControlFunction
)


def test_sigmoid_inverse():
    """Test that sigmoid and inverse_sigmoid are proper inverses."""
    print("Testing sigmoid and inverse_sigmoid...")

    # Test values in (0, 1)
    y_values = [0.1, 0.5, 0.9, 0.99, 0.01]
    for y in y_values:
        x = inverse_sigmoid(y)
        y_recovered = sigmoid(x)
        print(f"  y={y:.6f} -> x={x:.6f} -> y_recovered={y_recovered:.6f}")
        assert np.abs(y - y_recovered) < EPSILON

    print("  ✓ Sigmoid inverse test passed\n")


def test_physical_transformations():
    """Test physical <-> normalized <-> sigmoid space transformations."""
    print("Testing physical value transformations...")

    value_min, value_max = 0.0, 1.0
    test_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    for value in test_values:
        # Physical -> normalized -> physical
        normalized = physical_to_normalized(value, value_min, value_max)
        value_recovered = normalized_to_physical(normalized, value_min, value_max)
        print(f"  value={value:.2f} -> normalized={normalized:.2f} -> value_recovered={value_recovered:.2f}")
        assert np.abs(value - value_recovered) < EPSILON

    print()

    # Test with different bounds
    value_min, value_max = 0.2, 0.5
    value = 0.35

    # Physical -> sigmoid space -> physical
    g = physical_to_sigmoid_space(value, value_min, value_max)
    value_recovered = sigmoid_space_to_physical(g, value_min, value_max)
    print(f"  With bounds [{value_min}, {value_max}]:")
    print(f"  value={value:.2f} -> g={g:.6f} -> value_recovered={value_recovered:.6f}")
    assert np.abs(value - value_recovered) < EPSILON

    print("  ✓ Physical transformation test passed\n")


def test_coefficient_bounds():
    """Test coefficient bounds calculation."""
    print("Testing coefficient bounds calculation...")

    # Test different epsilon values explicitly
    eps_values = [LOOSE_EPSILON, 1e-8, 1e-6]
    for eps in eps_values:
        lower, upper = calculate_coefficient_bounds(eps)

        # Verify sigmoid at bounds gives eps and 1-eps
        sig_lower = sigmoid(lower)
        sig_upper = sigmoid(upper)

        print(f"  eps={eps:.2e}:")
        print(f"    Coefficient bounds: [{lower:.4f}, {upper:.4f}]")
        print(f"    sigmoid(lower)={sig_lower:.2e} (should be ≈ {eps:.2e})")
        print(f"    sigmoid(upper)={sig_upper:.6f} (should be ≈ {1-eps:.6f})")

        assert np.abs(sig_lower - eps) < eps * 0.01  # Within 1%
        assert np.abs(sig_upper - (1 - eps)) < eps * 0.01

    print("  ✓ Coefficient bounds test passed\n")


def test_constant_function_initialization():
    """Test that create_initial_guess produces a constant function."""
    print("Testing constant function initialization...")

    t_start, t_end = 0.0, 400.0
    n_basis = 5
    value_min, value_max = 0.0, 1.0
    initial_value = 0.5

    control = BasisControlFunction(
        t_start, t_end, n_basis,
        value_min=value_min, value_max=value_max,
        basis_type='chebyshev', eps=LOOSE_EPSILON
    )

    # Create initial guess
    coeffs = control.create_initial_guess(initial_value)

    print(f"  Initial value: {initial_value}")
    print(f"  Coefficients: {coeffs}")
    print(f"    c_0 = {coeffs[0]:.6f}")
    print(f"    Other coefficients: {coeffs[1:]}")

    # Verify all other coefficients are zero
    assert np.allclose(coeffs[1:], 0.0)

    # Verify value(t) = initial_value at various times
    test_times = np.linspace(t_start, t_end, 10)
    values = [control.evaluate(t, coeffs) for t in test_times]

    print(f"  Testing at {len(test_times)} time points:")
    for i, (t, value) in enumerate(zip(test_times, values)):
        if i < 3 or i >= len(test_times) - 1:  # Print first 3 and last
            print(f"    t={t:6.1f}: value(t)={value:.6f} (error: {abs(value - initial_value):.2e})")
        elif i == 3:
            print(f"    ...")
        assert np.abs(value - initial_value) < EPSILON

    print("  ✓ Constant function initialization test passed\n")


def test_bounds_enforcement():
    """Test that sigmoid transformation keeps values in bounds."""
    print("Testing bounds enforcement with extreme coefficients...")

    t_start, t_end = 0.0, 400.0
    n_basis = 3
    value_min, value_max = 0.2, 0.8

    control = BasisControlFunction(
        t_start, t_end, n_basis,
        value_min=value_min, value_max=value_max,
        basis_type='chebyshev', eps=LOOSE_EPSILON
    )

    # Get coefficient bounds
    coeff_lower, coeff_upper = control.get_coefficient_bounds()
    print(f"  Coefficient bounds: [{coeff_lower[0]:.4f}, {coeff_upper[0]:.4f}]")

    # Test with coefficients at bounds
    test_cases = [
        ("All at lower bound", np.full(n_basis, coeff_lower[0])),
        ("All at upper bound", np.full(n_basis, coeff_upper[0])),
        ("Mixed", np.array([coeff_upper[0], coeff_lower[0], 0.0])),
    ]

    test_times = np.linspace(t_start, t_end, 20)

    for name, coeffs in test_cases:
        values = control.evaluate(test_times, coeffs)
        value_min_actual = np.min(values)
        value_max_actual = np.max(values)

        print(f"  {name}:")
        print(f"    coeffs: {coeffs}")
        print(f"    value(t) range: [{value_min_actual:.6f}, {value_max_actual:.6f}]")
        print(f"    Expected: [{value_min:.6f}, {value_max:.6f}]")

        # Verify bounds are respected (within LOOSE_EPSILON tolerance)
        assert np.all(values >= value_min - LOOSE_EPSILON)
        assert np.all(values <= value_max + LOOSE_EPSILON)

    print("  ✓ Bounds enforcement test passed\n")


if __name__ == '__main__':
    print("="*80)
    print("BASIS FUNCTION TRANSFORMATION TESTS")
    print("="*80 + "\n")

    test_sigmoid_inverse()
    test_physical_transformations()
    test_coefficient_bounds()
    test_constant_function_initialization()
    test_bounds_enforcement()

    print("="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
