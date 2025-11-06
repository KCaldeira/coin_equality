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
# - Checking if values are effectively zero (e.g., fract_gdp ≈ 0)
# - Bounding variables away from exact 0 or 1 (e.g., Gini ∈ (ε, 1-ε))
# - Ensuring values stay strictly positive (e.g., A2 ≥ ε)
# - Root finding bracket offsets
EPSILON = 1e-12

# Looser epsilon for iterative convergence and optimization tolerances
# Used for:
# - Convergence criterion in y_eff iterative solver
# - Default value for xtol_abs in optimization (control parameter convergence)
# Provides practical precision (1e-8 absolute) without requiring machine precision
LOOSE_EPSILON = 1e-8

# Large value for detecting effectively infinite parameters
# Used for:
# - Checking if y_damage_halfsat is so large that damage is effectively uniform
# - Detecting when parameters should trigger special case handling
INVERSE_EPSILON = 1.0 / EPSILON

# Maximum iterations for initial capital stock convergence
# Used in integrate_model() to ensure convergence of K0 with climate damage
MAX_INITIAL_CAPITAL_ITERATIONS = 100
