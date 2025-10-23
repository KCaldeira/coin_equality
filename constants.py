"""
Numerical constants and tolerances for COIN_equality model.

Defines small and large values used for numerical stability and bounds checking.
Centralizes epsilon/bignum definitions to ensure consistency across the codebase.
"""

# Large negative number for utility when constraints are violated
# Used as penalty value when Gini or other variables are out of valid range
NEG_BIGNUM = -1e30

# Small epsilon for numerical comparisons and bounds
# Used for:
# - Comparing floats to unity (e.g., eta ≈ 1)
# - Checking if values are effectively zero (e.g., deltaL ≈ 0)
# - Bounding variables away from exact 0 or 1 (e.g., Gini ∈ (ε, 1-ε))
# - Ensuring values stay strictly positive (e.g., A2 ≥ ε)
# - Root finding bracket offsets
EPSILON = 1e-12
