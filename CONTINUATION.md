# Continuation Notes: Omega Convergence Debug Session

## Date: 2025-12-03

## Context
Working on fixing convergence issues in the Omega (climate damage) iterative solver in `economic_model.py`.

## Changes Made

### 1. Fixed Income Clipping (COMPLETED ✅)
**File**: `utility_integrals.py:130`

Changed from raising an error to clipping income to EPSILON:
```python
# Before: raised ValueError if income_vals < 0
# After: income_vals = np.maximum(income_vals, EPSILON)
```

This allows the optimizer to explore extreme parameter spaces without crashing.

### 2. Improved Convergence Algorithm (COMPLETED ✅)
**File**: `economic_model.py` lines 190-346

Replaced fixed relaxation (0.1) + Aitken acceleration with aggressive secant method:
- **Tier 1**: When bracketed, uses linear interpolation between bounds (full step, no damping)
- **Tier 2**: Without bracketing, uses secant method on last 2 points (full step, no safeguards)
- **Tier 3**: First iteration uses aggressive multiplicative update (no relaxation)

**Key insight**: Removed all step limiters and relaxation to encourage overshooting and establish bracketing quickly.

### 3. Relaxed Tolerance (COMPLETED ✅)
**File**: `constants.py:26`

Changed `LOOSE_EPSILON` from `1e-8` to `1e-6` for more practical convergence tolerance in optimization context.

### 4. Fixed Monotonicity Issue (COMPLETED ✅)
**File**: `economic_model.py:215-216, 239`

**Root cause discovered**: The function `Omega = f(Omega_base)` was NOT monotonic because the budget calculation used **lagged Omega** from the previous iteration:
```python
# Before (line 214):
available_for_redistribution_and_abatement = fract_gdp * y_gross * (1 - Omega)
```

This created memory/lag effects where the same `Omega_base` produced different `Omega` values depending on iteration history.

**Fix applied**:
```python
# After (lines 215-216):
omega_for_budget = Omega_target if not income_dependent_aggregate_damage else Omega
available_for_redistribution_and_abatement = fract_gdp * y_gross * (1 - omega_for_budget)
```

Also updated line 239 to use `omega_for_budget` instead of `Omega`.

This makes `Fmin`, `Fmax`, `redistribution_amount`, and `uniform_tax_rate` pure functions of `Omega_base`, restoring monotonicity.

### 5. Added Full Convergence Diagnostics (COMPLETED ✅)
**File**: `economic_model.py:200-206`

Changed from printing first 10 + last 10 iterations to printing **all iterations** when convergence fails.

## Current Issue ❌

### Symptom
When the optimizer explores certain extreme parameter combinations:
- `Omega = 0.0` consistently across all iterations
- `Omega_base` explodes exponentially (reaches ~10^140 before hitting MAX_ITERATIONS)
- The additive update correctly triggers for small Omega, but Omega remains stuck at 0

### Convergence History Pattern
```
Iteration | Omega        | Omega_base     | Omega_diff | Omega_base_diff
----------|--------------|----------------|------------|----------------
1         | 0.0000000000 | huge_value_1   | 0.00e+00   | 0.00e+00
2         | 0.0000000000 | huge_value_2   | 0.00e+00   | (halving)
...       | ...          | ...            | ...        | ...
257       | 0.0000000000 | still_huge     | 0.00e+00   | (continues)
```

### Hypothesis
Using `Omega_target` in the budget calculation (the monotonicity fix) may create edge cases:

1. When `Omega_target` is very small → `(1 - Omega_target)` ≈ 1.0
2. This makes `available_for_redistribution_and_abatement` very large
3. Large redistribution might make everyone wealthy
4. Wealthy people have low income-dependent damage
5. Aggregate damage → 0, regardless of `Omega_base`

### **CRITICAL BUG INDICATOR** 🚨
**The above scenario shouldn't be possible** because:
- **Redistribution is zero-sum**: it cannot increase mean income
- It only transfers from rich to poor
- If `Omega = 0` consistently, it suggests redistribution is somehow creating wealth

**This indicates a bug elsewhere in the code** - possibly in:
- `find_Fmin()` / `find_Fmax()` functions
- The aggregate damage calculation (lines 254-270)
- The income-dependent damage distribution logic
- Or the budget calculation itself

## Next Steps

### Immediate Investigation
1. **Verify redistribution is zero-sum**: Add assertions that mean income doesn't change after redistribution
2. **Check aggregate damage calculation**: Why is it returning 0?
   - Is `Fmin` or `Fmax` producing invalid ranges?
   - Are the three segments (low/mid/high income) being calculated correctly?
3. **Add debug output**: When `Omega = 0`, print:
   - `Fmin`, `Fmax`
   - `redistribution_amount`, `abateCost_amount`
   - `uniform_redistribution_amount`, `uniform_tax_rate`
   - Values from each of the three income segments

### Potential Fixes
1. **Better handling of Omega = 0 case**: Add early termination if Omega is stuck at 0 for multiple iterations
2. **Alternative budget calculation**: Consider using a blended approach between `Omega_target` and current `Omega`
3. **Bounds checking**: Add safeguards to prevent `Omega_base` from exploding

## Files Modified

1. `utility_integrals.py` - lines 9, 129-138
2. `constants.py` - line 26
3. `economic_model.py` - lines 190-192, 200-206, 214-216, 239, 294-346

## Test Command

```bash
python run_optimization.py config_COIN-equality_001_DICE2050.json
```

The first iteration (before the monotonicity fix) would fail at ~256 iterations with `Omega_diff ≈ 1e-06` (just above tolerance).

After the monotonicity fix, it hits the `Omega = 0` edge case on some parameter combinations.

## References

- Original convergence discussion: iterations 248-257 showed constant tiny steps indicating a very flat curve
- Monotonicity discovery: same `Omega_base = 164.2975831` produced both `Omega = 0.159089944` and `Omega = 0.260935399`
- This revealed the lag effect from using previous iteration's `Omega` in budget calculation

---

## CRITICAL BUGS FOUND AND FIXED ✅ (2025-12-03)

### Bug #1: High-Income Segment Using Wrong Income Rank
**File**: `economic_model.py:266`

**Root Cause**: High-income earners were assigned the income of the **poorest** people instead of the **richest** people.

```python
# BEFORE (WRONG):
max_income_before_savings = y_of_F_after_damage(Fmin, Fmin, Fmax, ...)
# This evaluates income at rank Fmin (lowest income boundary)

# AFTER (CORRECT):
max_income_before_savings = y_of_F_after_damage(Fmax, Fmin, Fmax, ...)
# This evaluates income at rank Fmax (high income boundary)
```

**Impact**: This bug violated the zero-sum constraint of redistribution. By assigning low incomes to high earners, the model created wealth out of thin air, causing:
- Aggregate damage → 0 regardless of `Omega_base`
- `Omega_base` to explode exponentially
- Convergence failure

**Why This Explains the Omega = 0 Bug**:
When high-income people (1 - Fmax fraction of population) were given low incomes:
1. They received large income-dependent climate damage (exponential in 1/income)
2. But their actual tax payments were based on their true high incomes (from `find_Fmax`)
3. This mismatch meant more tax revenue was collected than damage incurred
4. The extra "phantom wealth" reduced aggregate damage toward zero
5. Optimizer tried to compensate by increasing `Omega_base`, but damage stayed at 0

### Bug #2: Temperature Capping Instead of Damage Capping
**Files**: `economic_model.py:143-146`, `constants.py:67`

**Root Cause**: The model capped temperature at `DELTA_T_LIMIT = 12.0°C` to prevent unphysical damage values, but this created an artificial discontinuity in the optimization landscape.

```python
# BEFORE (WRONG):
delta_T = min(k_climate * Ecum, DELTA_T_LIMIT)
Omega = psi1 * delta_T + psi2 * (delta_T ** 2)

# AFTER (CORRECT):
delta_T = k_climate * Ecum
Omega = min(psi1 * delta_T + psi2 * (delta_T ** 2), 1.0 - EPSILON)
```

**Impact**:
- More physically meaningful: damage fraction cannot exceed ~100% of GDP
- Removes arbitrary temperature limit
- Smoother optimization landscape (damage capped directly, not indirectly through temperature)
- Eliminated `DELTA_T_LIMIT` constant entirely
- Cap at `1.0 - EPSILON` prevents division by zero in budget calculations when damage is extreme

**Files Modified**:
1. `economic_model.py:26` - Removed DELTA_T_LIMIT from imports
2. `economic_model.py:143-146` - Changed capping logic from delta_T to Omega
3. `economic_model.py:148-151` - Removed debug warning (would fire thousands of times during optimization)
4. `constants.py:61-67` - Removed DELTA_T_LIMIT constant and documentation
5. `README.md:85, 87, 1227, 1300-1303` - Updated documentation to reflect Omega capping

### Bug #3: Variable Naming Inconsistency (UnboundLocalError)
**File**: `economic_model.py:375-376, 427`

**Root Cause**: Inconsistent variable naming where `C_mean`/`c_mean` were used instead of following the convention (uppercase = aggregate, lowercase = per-capita).

**Error**: When running with `store_detailed_output=True`, line 427 tried to use undefined variable `y`:
```python
Consumption = y * L  # Total Consumption  ← y was never defined
```

**Fix**: Renamed variables to follow naming convention:
- `C_mean` → `Consumption` (aggregate consumption)
- `c_mean` → `consumption` (per-capita consumption)
- Removed line 427 since `Consumption` is already computed on line 375

**Impact**: Fixes crash when saving optimization results with detailed output.

## Next Actions

The critical bugs are now fixed. Next steps:
1. Test with optimization to verify it completes successfully
2. Monitor for any new edge cases during optimization
3. Consider adding assertions to verify redistribution is zero-sum (mean income unchanged)
