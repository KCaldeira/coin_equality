"""
Unit Test for Asymptotic Behavior of Gini Index Changes

This module validates the asymptotic behavior of the post-damage Gini coefficient
under climate damage with income-dependent damage function.

Test Cases:
1. ΔG → 0 as G₀ → 0 (perfect equality): No initial inequality to amplify
2. ΔG small when y_mean >> y_damage_halfsat: Wealthy population, minimal damage
3. ΔG increases with ω_max: Larger damage amplifies inequality more
4. ΔG larger with larger G₀ at y_mean = y_damage_halfsat: More initial inequality to amplify

Usage:
    python unit_test_gini_asymptotics.py
"""

from climate_damage_distribution import calculate_climate_damage_and_gini_effect


def test_gini_change_vanishes_with_perfect_equality():
    """
    Test 1: |ΔG| → 0 as G₀ → 0 (perfect equality)

    When initial Gini approaches 0, all individuals have nearly equal income.
    Even with regressive damage, there's no income spread to amplify, so
    the absolute change in Gini should approach zero.

    Physical interpretation: With perfect equality, everyone experiences
    similar damage, so inequality change (whether increase or decrease) → 0.

    Note: The sign of ΔG depends on the relationship between y_mean and
    y_damage_halfsat. What matters is |ΔG| → 0.
    """
    print("=" * 80)
    print("Test 1: ΔG → 0 as G₀ → 0 (Perfect Equality)")
    print("=" * 80)
    print("Expected: As initial Gini → 0, change in Gini (ΔG) → 0\n")

    # Set up parameters with regressive damage
    omega_max = 0.15  # 15% maximum damage
    y_damage_halfsat = 20000  # $20k half-saturation
    y_mean = 50000  # $50k mean income (well above half-sat, so damage is regressive)

    params = {
        'psi1': omega_max,
        'psi2': 0.0,
        'y_damage_halfsat': y_damage_halfsat
    }

    # Test with progressively smaller initial Gini (approaching perfect equality)
    Gini_initial_values = [0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

    print(f"  y_mean = ${y_mean:,}, y_damage_halfsat = ${y_damage_halfsat:,}, ω_max = {omega_max:.2%}\n")

    delta_G_values = []
    for Gini_initial in Gini_initial_values:
        _, Gini_climate = calculate_climate_damage_and_gini_effect(
            delta_T=1.0,
            Gini_current=Gini_initial,
            y_mean=y_mean,
            params=params
        )

        delta_G = Gini_climate - Gini_initial

        delta_G_values.append(delta_G)

        print(f"  G₀ = {Gini_initial:.4f}    G_climate = {Gini_climate:.6f}    ΔG = {delta_G:.6f}")

    # Check that |ΔG| is decreasing monotonically and approaching zero
    abs_delta_G_values = [abs(dG) for dG in delta_G_values]
    final_abs_delta_G = abs_delta_G_values[-1]
    tolerance = 0.001  # |ΔG| should be < 0.001 for very small G₀

    # Verify monotonic decrease in absolute value
    is_decreasing = all(abs_delta_G_values[i] >= abs_delta_G_values[i+1] for i in range(len(abs_delta_G_values)-1))

    if final_abs_delta_G < tolerance and is_decreasing:
        print(f"\n✓ PASS: |ΔG| → 0 as G₀ → 0 (final |ΔG| = {final_abs_delta_G:.6f}, decreasing monotonically)")
    else:
        print(f"\n✗ FAIL: |ΔG| = {final_abs_delta_G:.6f} (expected < {tolerance}), monotonic = {is_decreasing}")
        raise AssertionError("Test 1 failed: |ΔG| should approach 0 as G₀ → 0")

    print()


def test_gini_change_small_for_wealthy_population():
    """
    Test 2: |ΔG| small when y_mean >> y_damage_halfsat

    When mean income is much larger than half-saturation income, the entire
    population experiences very low damage (ω → 0 as y → ∞). With minimal
    damage, there's little scope for changing inequality.

    Physical interpretation: Wealthy populations experience negligible damage,
    so inequality cannot change significantly.
    """
    print("=" * 80)
    print("Test 2: ΔG Small for Wealthy Population (y_mean >> y_damage_halfsat)")
    print("=" * 80)
    print("Expected: When y_mean >> y_damage_halfsat, ΔG should be small\n")

    # Set up parameters
    omega_max = 0.20  # 20% maximum damage
    y_damage_halfsat = 1000  # $1k half-saturation
    Gini_initial = 0.4  # Moderate inequality

    params = {
        'psi1': omega_max,
        'psi2': 0.0,
        'y_damage_halfsat': y_damage_halfsat
    }

    # Test with progressively larger mean incomes (increasingly wealthy population)
    y_mean_values = [10000, 50000, 100000, 500000, 1000000]

    print(f"  y_damage_halfsat = ${y_damage_halfsat:,}, G₀ = {Gini_initial}, ω_max = {omega_max:.2%}\n")

    delta_G_values = []
    for y_mean in y_mean_values:
        _, Gini_climate = calculate_climate_damage_and_gini_effect(
            delta_T=1.0,
            Gini_current=Gini_initial,
            y_mean=y_mean,
            params=params
        )

        delta_G = Gini_climate - Gini_initial
        ratio = y_mean / y_damage_halfsat

        delta_G_values.append(delta_G)

        print(f"  y_mean/y_damage_halfsat = {ratio:8.1f}    ΔG = {delta_G:.6f}")

    # Check that |ΔG| is decreasing and final value is very small
    abs_delta_G_values = [abs(dG) for dG in delta_G_values]
    final_abs_delta_G = abs_delta_G_values[-1]
    tolerance = 0.001  # |ΔG| should be < 0.001 for very wealthy population

    # Verify monotonic decrease in absolute value
    is_decreasing = all(abs_delta_G_values[i] >= abs_delta_G_values[i+1] for i in range(len(abs_delta_G_values)-1))

    if final_abs_delta_G < tolerance and is_decreasing:
        print(f"\n✓ PASS: |ΔG| small for wealthy population (final |ΔG| = {final_abs_delta_G:.6f}, decreasing monotonically)")
    else:
        print(f"\n✗ FAIL: |ΔG| = {final_abs_delta_G:.6f} (expected < {tolerance}), monotonic = {is_decreasing}")
        raise AssertionError("Test 2 failed: |ΔG| should be small for wealthy populations")

    print()


def test_gini_change_increases_with_damage():
    """
    Test 3: |ΔG| increases with ω_max

    Larger maximum damage creates larger absolute differences in damage
    between rich and poor. Therefore, the magnitude of inequality change
    should increase monotonically with ω_max.

    Physical interpretation: Stronger climate damage changes income
    inequality more (in absolute magnitude).
    """
    print("=" * 80)
    print("Test 3: ΔG Increases with Damage Magnitude (ω_max)")
    print("=" * 80)
    print("Expected: ΔG should increase monotonically with ω_max\n")

    # Set up parameters with regressive damage
    # Use y_mean < y_damage_halfsat so damage amplifies inequality (regressive)
    y_damage_halfsat = 60000  # $60k half-saturation
    y_mean = 30000  # $30k mean income (poor population)
    Gini_initial = 0.4  # Moderate inequality

    # Test with progressively larger maximum damage
    omega_max_values = [0.05, 0.10, 0.15, 0.20, 0.25]

    print(f"  y_mean = ${y_mean:,}, y_damage_halfsat = ${y_damage_halfsat:,}, G₀ = {Gini_initial}\n")

    delta_G_values = []
    for omega_max in omega_max_values:
        params = {
            'psi1': omega_max,
            'psi2': 0.0,
            'y_damage_halfsat': y_damage_halfsat
        }

        _, Gini_climate = calculate_climate_damage_and_gini_effect(
            delta_T=1.0,
            Gini_current=Gini_initial,
            y_mean=y_mean,
            params=params
        )

        delta_G = Gini_climate - Gini_initial
        delta_G_values.append(delta_G)

        print(f"  ω_max = {omega_max:.2%}    ΔG = {delta_G:.6f}")

    # Verify monotonic increase in absolute value
    abs_delta_G_values = [abs(dG) for dG in delta_G_values]
    is_increasing = all(abs_delta_G_values[i] <= abs_delta_G_values[i+1] for i in range(len(abs_delta_G_values)-1))

    if is_increasing:
        print(f"\n✓ PASS: |ΔG| increases monotonically with ω_max")
    else:
        print(f"\n✗ FAIL: |ΔG| does not increase monotonically with ω_max")
        print(f"  |ΔG| values: {abs_delta_G_values}")
        raise AssertionError("Test 3 failed: |ΔG| should increase with ω_max")

    print()


def test_gini_change_larger_with_higher_initial_gini():
    """
    Test 4: |ΔG| larger with larger G₀ at y_mean = y_damage_halfsat

    At y_mean = y_damage_halfsat, the population spans the full range of the
    damage function. With higher initial Gini, there's a wider income spread,
    so the damage changes inequality more.

    Physical interpretation: Damage affects existing inequality.
    More initial inequality → larger magnitude of change.
    """
    print("=" * 80)
    print("Test 4: ΔG Increases with Initial Gini at Half-Saturation")
    print("=" * 80)
    print("Expected: At y_mean = y_damage_halfsat, ΔG should increase with G₀\n")

    # Set up parameters with mean income = half-saturation
    omega_max = 0.15  # 15% maximum damage
    y_damage_halfsat = 40000  # $40k half-saturation
    y_mean = y_damage_halfsat  # Mean equals half-saturation

    params = {
        'psi1': omega_max,
        'psi2': 0.0,
        'y_damage_halfsat': y_damage_halfsat
    }

    # Test with progressively larger initial Gini (more initial inequality)
    Gini_initial_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    print(f"  y_mean = y_damage_halfsat = ${y_mean:,}, ω_max = {omega_max:.2%}\n")

    delta_G_values = []
    for Gini_initial in Gini_initial_values:
        _, Gini_climate = calculate_climate_damage_and_gini_effect(
            delta_T=1.0,
            Gini_current=Gini_initial,
            y_mean=y_mean,
            params=params
        )

        delta_G = Gini_climate - Gini_initial
        delta_G_values.append(delta_G)

        print(f"  G₀ = {Gini_initial:.2f}    G_climate = {Gini_climate:.6f}    ΔG = {delta_G:.6f}")

    # Verify monotonic increase in absolute value
    abs_delta_G_values = [abs(dG) for dG in delta_G_values]
    is_increasing = all(abs_delta_G_values[i] <= abs_delta_G_values[i+1] for i in range(len(abs_delta_G_values)-1))

    if is_increasing:
        print(f"\n✓ PASS: |ΔG| increases monotonically with G₀ at y_mean = y_damage_halfsat")
    else:
        print(f"\n✗ FAIL: |ΔG| does not increase monotonically with G₀")
        print(f"  |ΔG| values: {abs_delta_G_values}")
        raise AssertionError("Test 4 failed: |ΔG| should increase with G₀ when y_mean = y_damage_halfsat")

    print()


def run_all_gini_asymptotic_tests():
    """Run all Gini asymptotic behavior tests."""
    print("\n" + "=" * 80)
    print("GINI INDEX ASYMPTOTIC BEHAVIOR VALIDATION")
    print("=" * 80)
    print()

    try:
        test_gini_change_vanishes_with_perfect_equality()
        test_gini_change_small_for_wealthy_population()
        test_gini_change_increases_with_damage()
        test_gini_change_larger_with_higher_initial_gini()

        print("=" * 80)
        print("ALL GINI ASYMPTOTIC TESTS PASSED")
        print("=" * 80)

    except AssertionError as e:
        print("=" * 80)
        print(f"GINI ASYMPTOTIC TEST FAILED: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    run_all_gini_asymptotic_tests()
