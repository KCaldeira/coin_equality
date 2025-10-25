"""
Unit Tests for Section 4: Redistribution Mechanics

This module validates the analytical solutions for income redistribution mechanics,
including crossing rank calculations, inverse problem solving (G2 from deltaL),
and effective Gini under mixed redistribution/abatement allocation.

Key equations tested:
- Eq. (4.1): Fraction of income redistributed (inverse problem G2 from deltaL)
- Eq. (4.2): Crossing rank F* where income remains unchanged
- Eq. (4.4): Effective Gini index under mixed allocation

The tests verify:
1. Crossing rank satisfies Lorenz derivative equality (equal slopes)
2. Crossing rank monotonicity properties
3. G2_from_deltaL caps correctly and round-trips
4. Effective Gini calculation maintains invariants
5. Crossing rank preservation at f=0

Usage:
    python unit_test_eq4.x.py

All tests will run automatically and report PASS/FAIL status.
"""

import math
import random
import numpy as np

from income_distribution import (
    a_from_G,
    L_pareto,
    L_pareto_derivative,
    crossing_rank_from_G,
    G2_from_deltaL,
    calculate_Gini_effective_redistribute_abate,
)


# ============================================================================
# Helper Functions
# ============================================================================

def A_from_G(G):
    """
    Convenience parameter A = (1-G)/(1+G).

    This appears in various redistribution formulas and simplifies
    the relationship between different Gini indices.

    Parameters
    ----------
    G : float
        Gini index (0 < G < 1)

    Returns
    -------
    float
        Parameter A
    """
    return (1.0 - G) / (1.0 + G)


def phi(r):
    """
    Helper function φ(r) = (r - 1) · r^{1/(r-1) - 1}

    This function appears in the relationship between deltaL and the
    ratio of Gini parameters. Robust evaluation across r<1, r≈1, r>1.

    Parameters
    ----------
    r : float
        Ratio parameter

    Returns
    -------
    float
        φ(r) value
    """
    if r <= 0:
        return float("-inf")
    if abs(r - 1.0) < 1e-12:
        return 0.0
    sgn = 1.0 if r > 1.0 else -1.0
    log_abs = math.log(abs(r - 1.0)) + (1.0 / (r - 1.0) - 1.0) * math.log(r)
    return sgn * math.exp(log_abs)


# ============================================================================
# Test 1: Crossing Rank Satisfies Lorenz Derivative Equality
# ============================================================================

def test_crossing_rank_satisfies_equation():
    """
    Test that the crossing rank F* satisfies dL/dF(F*, G1) = dL/dF(F*, G2).

    The crossing rank is the population percentile where the Lorenz curves
    have equal slopes, meaning income at rank F remains unchanged during
    redistribution from Gini G1 to G2.
    """
    print("=" * 80)
    print("Test 1: Crossing Rank Satisfies Lorenz Derivative Equality")
    print("=" * 80)
    print("\nVerifying that dL/dF(F*, G1) = dL/dF(F*, G2) at crossing rank F*\n")

    test_cases = [
        (0.6, 0.3),  # Large reduction in inequality
        (0.3, 0.6),  # Increase in inequality
        (0.5, 0.5),  # Equal Gini -> crossing at 0.5
        (0.2, 0.1),  # Small reduction
    ]

    all_passed = True
    for i, (G1, G2) in enumerate(test_cases, 1):
        Fstar = crossing_rank_from_G(G1, G2)
        dL1 = L_pareto_derivative(Fstar, G1)
        dL2 = L_pareto_derivative(Fstar, G2)
        error = abs(dL1 - dL2)

        passed = error < 1e-10 and 0.0 <= Fstar <= 1.0
        all_passed = all_passed and passed

        print(f"Case {i}: G1={G1:.1f} → G2={G2:.1f}")
        print(f"         F* = {Fstar:.6f}")
        print(f"         dL/dF(F*, G1) = {dL1:.10f}")
        print(f"         dL/dF(F*, G2) = {dL2:.10f}")
        print(f"         Error = {error:.2e}  {'✓ PASS' if passed else '✗ FAIL'}")
        print()

    print(f"Result: {'All cases PASSED' if all_passed else 'Some cases FAILED'}")
    print("=" * 80)
    print()
    return all_passed


# ============================================================================
# Test 2: Crossing Rank Monotonicity
# ============================================================================

def test_crossing_rank_monotonic_direction():
    """
    Test that crossing rank F* is monotone in G2.

    Property: F*(G1, G2) should be monotonically increasing in G2.
    - As G2 decreases (more redistribution): F* decreases
    - As G2 increases (less redistribution): F* increases
    """
    print("=" * 80)
    print("Test 2: Crossing Rank Monotonicity")
    print("=" * 80)
    print("\nVerifying F* is monotonically increasing in G2\n")

    G1 = 0.6
    all_passed = True

    # Create a sequence of G2 values from low to high
    G2_values = np.linspace(0.1, 0.95, 40)
    F_stars = [crossing_rank_from_G(G1, G2) for G2 in G2_values]

    # Check monotonicity: F* should increase as G2 increases
    monotone_failures = []
    for i in range(len(F_stars) - 1):
        if F_stars[i + 1] < F_stars[i] - 1e-12:  # Allow small numerical tolerance
            monotone_failures.append((G2_values[i], G2_values[i+1], F_stars[i], F_stars[i+1]))
            all_passed = False

    if all_passed:
        print(f"  ✓ Monotonicity verified: F* increases with G2")
        print(f"    G2 range: [{G2_values[0]:.2f}, {G2_values[-1]:.2f}]")
        print(f"    F* range: [{F_stars[0]:.6f}, {F_stars[-1]:.6f}]")
        print(f"    Tested {len(G2_values)} values")
    else:
        print(f"  ✗ Monotonicity FAILED at {len(monotone_failures)} points:")
        for G2_a, G2_b, F_a, F_b in monotone_failures[:5]:  # Show first 5 failures
            print(f"    G2: {G2_a:.4f} → {G2_b:.4f}, F*: {F_a:.6f} → {F_b:.6f} (decreased!)")
        if len(monotone_failures) > 5:
            print(f"    ... and {len(monotone_failures) - 5} more failures")

    # Also test specific boundary behavior
    print("\n  Boundary checks:")
    print(f"    G1=G2={G1}: F*={crossing_rank_from_G(G1, G1):.6f} (should be 0.5)")
    F_at_equal = crossing_rank_from_G(G1, G1)
    boundary_passed = abs(F_at_equal - 0.5) < 1e-10
    print(f"    {'✓ PASS' if boundary_passed else '✗ FAIL'}")
    all_passed = all_passed and boundary_passed

    print(f"\nResult: {'All monotonicity tests PASSED' if all_passed else 'Some tests FAILED'}")
    print("=" * 80)
    print()
    return all_passed


# ============================================================================
# Test 3: G2_from_deltaL Caps and Round-Trip
# ============================================================================

def test_G2_from_deltaL_caps_and_roundtrip():
    """
    Test the inverse problem: finding G2 from deltaL (fraction redistributed).

    Verifies:
    1. Below the cap: solution round-trips (deltaL → G2 → deltaL)
    2. Above the cap: solution clamps to G2=0 and returns remainder
    """
    print("=" * 80)
    print("Test 3: G2 from deltaL - Caps and Round-Trip")
    print("=" * 80)
    print("\nTesting inverse problem: given deltaL, find target Gini G2")
    print("Verifying round-trip accuracy and cap behavior\n")

    random.seed(0)
    all_passed = True
    max_roundtrip_error = 0.0
    n_cases = 50

    for case_num in range(1, n_cases + 1):
        G1 = random.uniform(0.05, 0.9)
        A1 = A_from_G(G1)
        r_max = 1.0 / A1  # Corresponds to G2 → 0
        deltaL_max = phi(r_max)

        # Test 1: Below the cap - should round-trip
        deltaL = random.uniform(-0.25, 0.999) * (deltaL_max - 1e-12)
        G2, rem = G2_from_deltaL(deltaL, G1)

        passed_roundtrip = rem == 0.0

        # Verify round-trip through φ relation
        A2 = A_from_G(G2)
        r = A2 / A1
        deltaL_implied = phi(r)
        roundtrip_error = abs(deltaL_implied - deltaL) / max(1.0, abs(deltaL))
        max_roundtrip_error = max(max_roundtrip_error, roundtrip_error)

        passed_roundtrip = passed_roundtrip and roundtrip_error <= 1e-10

        # Test 2: Above the cap - should clamp to G2=0
        overshoot = random.uniform(1e-6, 1e-2)
        deltaL_big = deltaL_max + overshoot
        G2_cap, rem_cap = G2_from_deltaL(deltaL_big, G1)

        passed_cap = (G2_cap == 0.0) and (abs(rem_cap - overshoot) <= 1e-10)

        passed = passed_roundtrip and passed_cap
        all_passed = all_passed and passed

        if case_num <= 5 or not passed:  # Show first 5 cases and any failures
            print(f"Case {case_num:2d}: G1={G1:.4f}, deltaL_max={deltaL_max:.6f}")
            print(f"         Round-trip: deltaL={deltaL:.6f} → G2={G2:.4f} → "
                  f"deltaL={deltaL_implied:.6f}, error={roundtrip_error:.2e}")
            print(f"         Cap test: deltaL={deltaL_big:.6f} → G2={G2_cap:.4f}, "
                  f"remainder={rem_cap:.6f}")
            print(f"         {'✓ PASS' if passed else '✗ FAIL'}")
            print()

    if n_cases > 5:
        print(f"... {n_cases - 5} additional cases tested ...")
        print()

    print(f"Maximum round-trip error: {max_roundtrip_error:.2e}")
    print(f"Result: {'All {n_cases} cases PASSED' if all_passed else 'Some cases FAILED'}")
    print("=" * 80)
    print()
    return all_passed


# ============================================================================
# Test 4: Effective Gini Calculation Invariants
# ============================================================================

def test_calculate_Gini_effective_redistribute_abate_invariants():
    """
    Test the effective Gini calculation under mixed allocation.

    When fraction f goes to abatement (1-f to redistribution), verifies:
    1. f=0: full redistribution → G_eff = G2_full (minimum Gini)
    2. f=1: no redistribution → G_eff close to G1 (maximum Gini)
    3. Monotonicity: as f increases, G_eff increases (less equality)

    IMPORTANT: deltaL must be < max_F |L(F,G1) - L(F,G2)|
    """
    print("=" * 80)
    print("Test 4: Effective Gini Calculation - Invariants")
    print("=" * 80)
    print("\nTesting Eq. (4.4): effective Gini under mixed allocation")
    print("Verifying boundary conditions and monotonicity")
    print("Using deltaL values that respect Lorenz curve separation constraint\n")

    random.seed(1)
    all_passed = True
    n_cases = 30

    for case_num in range(1, n_cases + 1):
        G1 = random.uniform(0.1, 0.8)

        # Use small deltaL that will respect Lorenz curve separation
        # For Pareto distributions, max separation ≈ (G1 - G2)
        # Use conservative deltaL ≈ 0.2 * (G1 - G2) to stay well within bounds
        target_G2 = G1 * random.uniform(0.5, 0.9)  # Reduce G1 by 10-50%
        deltaL = 0.15 * abs(G1 - target_G2)  # Conservative, well within separation

        # Full-redistribution target
        G2_full, rem = G2_from_deltaL(deltaL, G1)
        passed = rem == 0.0

        # Test f=0 (full redistribution)
        G2_eff_0, rem0 = calculate_Gini_effective_redistribute_abate(0.0, deltaL, G1)
        passed = passed and rem0 == 0.0 and abs(G2_eff_0 - G2_full) <= 1e-12

        # Test f=1 (no redistribution, all to abatement)
        G2_eff_1, rem1 = calculate_Gini_effective_redistribute_abate(1.0, deltaL, G1)
        passed = passed and rem1 == 0.0 and G2_full <= G2_eff_1 <= G1 + 1e-12

        # Test monotonicity: G_eff should increase with f
        f_vals = np.linspace(0.0, 1.0, 11)
        Gs = [calculate_Gini_effective_redistribute_abate(f, deltaL, G1)[0] for f in f_vals]
        monotone = all(Gs[i] <= Gs[i + 1] + 1e-10 for i in range(len(Gs) - 1))
        passed = passed and monotone

        all_passed = all_passed and passed

        if case_num <= 5 or not passed:  # Show first 5 cases and any failures
            print(f"Case {case_num:2d}: G1={G1:.4f}, deltaL={deltaL:.6f}")
            print(f"         G2_full={G2_full:.4f} (f=0 target)")
            print(f"         G_eff(f=0)={G2_eff_0:.4f}, error={abs(G2_eff_0 - G2_full):.2e}")
            print(f"         G_eff(f=1)={G2_eff_1:.4f} (should be ≥ G2_full, ≤ G1)")
            print(f"         Monotone: {monotone}")
            print(f"         {'✓ PASS' if passed else '✗ FAIL'}")
            print()

    if n_cases > 5:
        print(f"... {n_cases - 5} additional cases tested ...")
        print()

    print(f"Result: {'All {n_cases} cases PASSED' if all_passed else 'Some cases FAILED'}")
    print("=" * 80)
    print()
    return all_passed


# ============================================================================
# Test 5: Crossing Rank Preservation at f=0
# ============================================================================

def test_calculate_Gini_effective_crossing_invariance():
    """
    Test crossing rank behavior at f=0.

    At f=0 (full redistribution), the crossing rank should be preserved:
    F*(G1, G_eff) = F*(G1, G2_full) because G_eff = G2_full.

    At f>0, the crossing rank will differ as less redistribution occurs.
    """
    print("=" * 80)
    print("Test 5: Effective Gini - Crossing Rank at f=0")
    print("=" * 80)
    print("\nVerifying that F* is preserved only at f=0 (full redistribution)")
    print("At f>0, F* is allowed to vary\n")

    G1 = 0.6
    # Use smaller deltaL that respects Lorenz curve separation constraint
    deltaL = 0.02
    G2_full, _ = G2_from_deltaL(deltaL, G1)
    Fstar_target = crossing_rank_from_G(G1, G2_full)

    print(f"Parameters: G1={G1}, deltaL={deltaL}")
    print(f"Full redistribution: G2_full={G2_full:.4f}")
    print(f"Target crossing rank: F*={Fstar_target:.6f}\n")

    all_passed = True
    f_vals = np.linspace(0.0, 1.0, 6)

    for f in f_vals:
        G2_eff, _ = calculate_Gini_effective_redistribute_abate(f, deltaL, G1)
        Fstar_eff = crossing_rank_from_G(G1, G2_eff)
        error = abs(Fstar_eff - Fstar_target)

        if f == 0.0:
            # At f=0, crossing rank should be preserved
            passed = error <= 1e-10
            status = f"{'✓ PASS' if passed else '✗ FAIL'} (preserved at f=0)"
        else:
            # At f>0, crossing rank is allowed to differ
            passed = True  # No constraint
            status = "✓ (allowed to vary)"

        all_passed = all_passed and passed

        print(f"f={f:.2f}: G_eff={G2_eff:.4f}, F*(G1,G_eff)={Fstar_eff:.6f}, "
              f"error={error:.2e}  {status}")

    print(f"\nResult: {'All cases PASSED' if all_passed else 'Some cases FAILED'}")
    print("=" * 80)
    print()
    return all_passed


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """
    Run all unit tests for redistribution mechanics.

    Returns
    -------
    bool
        True if all tests passed, False otherwise
    """
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  Unit Tests: Section 4 - Redistribution Mechanics".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    results = []

    # Run all tests
    results.append(("Crossing Rank Derivative Equality", test_crossing_rank_satisfies_equation()))
    results.append(("Crossing Rank Monotonicity", test_crossing_rank_monotonic_direction()))
    results.append(("G2 from deltaL Round-Trip", test_G2_from_deltaL_caps_and_roundtrip()))
    results.append(("Effective Gini Invariants", test_calculate_Gini_effective_redistribute_abate_invariants()))
    results.append(("Crossing Rank at f=0", test_calculate_Gini_effective_crossing_invariance()))

    # Summary
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  Test Summary".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    for i, (name, passed) in enumerate(results, 1):
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {i}. {name:<50} {status}")

    print()
    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("  " + "=" * 76)
        print(f"  {'ALL TESTS PASSED':^76}")
        print("  " + "=" * 76)
    else:
        print("  " + "=" * 76)
        print(f"  {'SOME TESTS FAILED':^76}")
        print("  " + "=" * 76)

    print()
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
