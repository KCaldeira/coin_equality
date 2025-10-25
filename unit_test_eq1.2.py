"""
Unit Test for Equation (1.2): Climate Damage Analytical Solution

This module validates the analytical solution for aggregate climate damage (Ω)
by comparing it against high-precision numerical integration.

The analytical solution uses the hypergeometric function ₂F₁:
    Ω = ω_max · (k/ȳ) · ₂F₁(1, a, a+1, -β)
    where β = k·a/(ȳ·(a-1))

This test verifies that the analytical formula matches numerical integration
to within 1e-9 relative tolerance across a wide range of parameter values.

Usage:
    python unit_test_eq1.2.py

The test will print a summary of each case and report PASS/FAIL status.
"""

import mpmath as mp
import random
from climate_damage_distribution import calculate_climate_damage_and_gini_effect
from income_distribution import a_from_G

# Set high precision for numerical integration (80 decimal places)
mp.mp.dps = 80


def y_of_F(F, a, ybar):
    """
    Pareto income distribution: income as a function of population rank F.

    Parameters
    ----------
    F : float
        Population fraction (0 = poorest, 1 = richest)
    a : float
        Pareto parameter (a > 1)
    ybar : float
        Mean income

    Returns
    -------
    float
        Income at population rank F
    """
    return ybar * (1 - 1/a) * (1 - F)**(-1/a)


def omega_y(y, omega_max, k):
    """
    Half-saturation damage function: damage fraction as a function of income.

    Parameters
    ----------
    y : float
        Income level
    omega_max : float
        Maximum damage fraction (at y=0)
    k : float
        Half-saturation income (damage = omega_max/2 at y=k)

    Returns
    -------
    float
        Damage fraction at income y
    """
    return omega_max * k / (k + y)


def omega_numeric(a, ybar, omega_max, k):
    """
    Numerically integrate aggregate damage using high-precision mpmath.

    This computes the "ground truth" value of Ω via numerical integration:
        Ω = (1/ȳ) · ∫₀¹ ω(y(F)) · y(F) · dF

    Parameters
    ----------
    a : float
        Pareto parameter
    ybar : float
        Mean income
    omega_max : float
        Maximum damage fraction
    k : float
        Half-saturation income

    Returns
    -------
    float
        Numerically integrated aggregate damage fraction
    """
    integrand = lambda F: omega_y(y_of_F(F, a, ybar), omega_max, k) * y_of_F(F, a, ybar)
    num = mp.quad(integrand, [0, 1])
    return float(num / ybar)


def test_eq12_random_cases():
    """
    Test analytical solution against numerical integration for random parameters.

    Generates 10 random test cases with varying:
    - Gini index (inequality level)
    - Mean income
    - Half-saturation income
    - Maximum damage fraction

    Each case verifies that the analytical solution matches numerical integration
    to within 1e-9 relative tolerance.
    """
    print("=" * 80)
    print("Unit Test: Equation (1.2) - Climate Damage Analytical Solution")
    print("=" * 80)
    print("\nValidating analytical hypergeometric solution against numerical integration")
    print("Target tolerance: 1e-9 relative error\n")

    random.seed(0)
    max_rel_error = 0.0

    for i in range(10):
        # Generate random parameters
        G = random.uniform(0.2, 0.7)           # Gini index
        a = a_from_G(G)                        # Pareto parameter
        ybar = 10**random.uniform(3, 5)        # Mean income: ~1e3 to 1e5
        k = 10**random.uniform(0, 4)           # Half-saturation: ~1 to 1e4
        omega_max = random.uniform(0.05, 0.3)  # Max damage: 5% to 30%

        # Set up parameters for the function
        # (delta_T=1.0 and k_damage_exp=1.0 means omega_max = k_damage_coeff)
        params = dict(
            k_damage_halfsat=k,
            k_damage_coeff=omega_max,
            k_damage_exp=1.0
        )

        # Call analytical solution (function under test)
        Omega_analytical, _ = calculate_climate_damage_and_gini_effect(
            delta_T=1.0,
            Gini_current=G,
            y_mean=ybar,
            params=params
        )

        # Compute numerical ground truth using high-precision integration
        Omega_numerical = omega_numeric(a, ybar, omega_max, k)

        # Calculate relative error
        rel_error = abs(Omega_analytical - Omega_numerical) / max(1.0, Omega_numerical)
        max_rel_error = max(max_rel_error, rel_error)

        # Check against tolerance
        tolerance = 1e-9
        passed = rel_error <= tolerance

        # Print results for this case
        print(f"Case {i+1:2d}:  G={G:.4f}  ȳ={ybar:8.1f}  k={k:8.1f}  ω_max={omega_max:.4f}")
        print(f"          Ω_analytical = {Omega_analytical:.12f}")
        print(f"          Ω_numerical  = {Omega_numerical:.12f}")
        print(f"          Rel. error   = {rel_error:.2e}  {'✓ PASS' if passed else '✗ FAIL'}")
        print()

        # Assert to catch failures
        assert passed, f"Test case {i+1} failed: relative error {rel_error:.2e} exceeds tolerance {tolerance:.2e}"

    print("=" * 80)
    print(f"All 10 test cases PASSED")
    print(f"Maximum relative error: {max_rel_error:.2e}")
    print("=" * 80)


if __name__ == "__main__":
    test_eq12_random_cases()
