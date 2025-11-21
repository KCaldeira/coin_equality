"""
Unit Test for Edge Cases of Climate Damage Function

This module validates the edge case behavior of the climate damage analytical solution.

Test Cases:
1. When y_mean << y_damage_distribution_scale, Ω should approach ω_max (poor population)
2. When y_mean >> y_damage_distribution_scale, Ω should approach 0 (wealthy population)
3. When Gini → 0 and y_mean = y_damage_distribution_scale, Ω should approach ω_max/2 (equal incomes at half-saturation)

Usage:
    python unit_test_edge_cases.py
"""

from climate_damage_distribution import calculate_climate_damage_and_gini_effect


def test_edge_case_1_poor_population():
    """
    Test Case 1: y_mean << y_damage_distribution_scale → Ω ≈ ω_max

    When mean income is very small compared to half-saturation income,
    the entire population is in the high-damage regime where ω(y) ≈ ω_max.
    Therefore aggregate damage should approach ω_max.
    """
    print("=" * 80)
    print("Edge Case 1: Poor Population (y_mean << y_damage_distribution_scale)")
    print("=" * 80)
    print("Expected: Ω → ω_max\n")

    # Set up parameters
    omega_max = 0.1  # 10% maximum damage
    y_damage_distribution_scale = 100000  # $100k half-saturation

    params = {
        'psi1': omega_max,
        'psi2': 0.0,
        'y_damage_distribution_scale': y_damage_distribution_scale
    }

    # Test with progressively smaller mean incomes
    y_mean_values = [10000, 1000, 100, 10]  # Much smaller than y_damage_distribution_scale
    Gini = 0.4  # Moderate inequality

    for y_mean in y_mean_values:
        Omega, _ = calculate_climate_damage_and_gini_effect(
            delta_T=1.0,
            Gini_current=Gini,
            y_mean=y_mean,
            params=params
        )

        ratio = y_mean / y_damage_distribution_scale
        error = abs(Omega - omega_max) / omega_max

        print(f"  y_mean/y_damage_distribution_scale = {ratio:8.2e}    Ω = {Omega:.6f}    ω_max = {omega_max:.6f}    error = {error:.2e}")

    # For the smallest ratio, Omega should be very close to omega_max
    final_error = abs(Omega - omega_max) / omega_max
    tolerance = 0.1  # 10% tolerance for this extreme case

    if final_error < tolerance:
        print(f"\n✓ PASS: For y_mean << y_damage_distribution_scale, Ω ≈ ω_max (error = {final_error:.2e})")
    else:
        print(f"\n✗ FAIL: Error {final_error:.2e} exceeds tolerance {tolerance:.2e}")
        raise AssertionError(f"Edge case 1 failed: Ω should approach ω_max")

    print()


def test_edge_case_2_wealthy_population():
    """
    Test Case 2: y_mean >> y_damage_distribution_scale → Ω ≈ 0

    When mean income is very large compared to half-saturation income,
    the entire population is in the low-damage regime where ω(y) ≈ 0.
    Therefore aggregate damage should approach 0.
    """
    print("=" * 80)
    print("Edge Case 2: Wealthy Population (y_mean >> y_damage_distribution_scale)")
    print("=" * 80)
    print("Expected: Ω → 0\n")

    # Set up parameters
    omega_max = 0.1  # 10% maximum damage
    y_damage_distribution_scale = 1000  # $1k half-saturation

    params = {
        'psi1': omega_max,
        'psi2': 0.0,
        'y_damage_distribution_scale': y_damage_distribution_scale
    }

    # Test with progressively larger mean incomes
    y_mean_values = [10000, 100000, 1000000, 10000000]  # Much larger than y_damage_distribution_scale
    Gini = 0.4  # Moderate inequality

    for y_mean in y_mean_values:
        Omega, _ = calculate_climate_damage_and_gini_effect(
            delta_T=1.0,
            Gini_current=Gini,
            y_mean=y_mean,
            params=params
        )

        ratio = y_mean / y_damage_distribution_scale

        print(f"  y_mean/y_damage_distribution_scale = {ratio:8.2e}    Ω = {Omega:.6e}")

    # For the largest ratio, Omega should be very close to 0
    tolerance = 1e-3  # Omega should be < 0.001

    if Omega < tolerance:
        print(f"\n✓ PASS: For y_mean >> y_damage_distribution_scale, Ω → 0 (Ω = {Omega:.2e})")
    else:
        print(f"\n✗ FAIL: Ω = {Omega:.2e} exceeds tolerance {tolerance:.2e}")
        raise AssertionError(f"Edge case 2 failed: Ω should approach 0")

    print()


def test_edge_case_3_equal_incomes_at_halfsat():
    """
    Test Case 3: Gini → 0 and y_mean = y_damage_distribution_scale → Ω ≈ ω_max/2

    When Gini approaches 0, we have perfect equality (all incomes equal).
    When all incomes equal y_damage_distribution_scale, everyone experiences damage:
        ω(y_damage_distribution_scale) = ω_max · y_damage_distribution_scale / (y_damage_distribution_scale + y_damage_distribution_scale)
                             = ω_max / 2
    Therefore aggregate damage should equal ω_max/2.
    """
    print("=" * 80)
    print("Edge Case 3: Equal Incomes at Half-Saturation (Gini → 0, y_mean = y_damage_distribution_scale)")
    print("=" * 80)
    print("Expected: Ω → ω_max/2\n")

    # Set up parameters
    omega_max = 0.1  # 10% maximum damage
    y_damage_distribution_scale = 50000  # $50k half-saturation
    y_mean = y_damage_distribution_scale  # Mean income equals half-saturation

    params = {
        'psi1': omega_max,
        'psi2': 0.0,
        'y_damage_distribution_scale': y_damage_distribution_scale
    }

    # Test with progressively smaller Gini (approaching perfect equality)
    Gini_values = [0.3, 0.2, 0.1, 0.05, 0.01]
    expected = omega_max / 2.0

    for Gini in Gini_values:
        Omega, _ = calculate_climate_damage_and_gini_effect(
            delta_T=1.0,
            Gini_current=Gini,
            y_mean=y_mean,
            params=params
        )

        error = abs(Omega - expected) / expected

        print(f"  Gini = {Gini:.3f}    Ω = {Omega:.6f}    ω_max/2 = {expected:.6f}    error = {error:.2e}")

    # For the smallest Gini, Omega should be very close to omega_max/2
    final_error = abs(Omega - expected) / expected
    tolerance = 0.01  # 1% tolerance

    if final_error < tolerance:
        print(f"\n✓ PASS: For Gini → 0 and y_mean = y_damage_distribution_scale, Ω ≈ ω_max/2 (error = {final_error:.2e})")
    else:
        print(f"\n✗ FAIL: Error {final_error:.2e} exceeds tolerance {tolerance:.2e}")
        raise AssertionError(f"Edge case 3 failed: Ω should approach ω_max/2")

    print()


def run_all_edge_case_tests():
    """Run all edge case tests."""
    print("\n" + "=" * 80)
    print("EDGE CASE VALIDATION FOR CLIMATE DAMAGE FUNCTION")
    print("=" * 80)
    print()

    try:
        test_edge_case_1_poor_population()
        test_edge_case_2_wealthy_population()
        test_edge_case_3_equal_incomes_at_halfsat()

        print("=" * 80)
        print("ALL EDGE CASE TESTS PASSED")
        print("=" * 80)

    except AssertionError as e:
        print("=" * 80)
        print(f"EDGE CASE TEST FAILED: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    run_all_edge_case_tests()
