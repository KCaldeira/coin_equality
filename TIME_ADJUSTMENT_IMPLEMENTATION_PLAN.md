# Time Adjustment Optimization - Implementation Plan

## Overview

After the standard iterative refinement optimization completes, add one additional optimization phase that adjusts the **timing** of control points while keeping the **control values** fixed. This allows the optimizer to find the optimal temporal spacing of interventions.

## Problem Statement

Given N control points from the final iteration:
- Times: `[t₀, t₁, t₂, ..., t_{N-1}]`
- Values: `[f₀, f₁, f₂, ..., f_{N-1}]`

Where:
- `t₀ = t_start` (fixed)
- `t_{N-1} = t_end` (fixed)

**Goal**: Optimize the interior time points `t₁, t₂, ..., t_{N-2}` to maximize the objective function while keeping all f values constant.

## Parameterization

For each interior point n (where n = 1 to N-2):

```
t_new[n] = ctrl[n-1] * (t[n+1] - t[n-1]) + t[n-1]
```

Where:
- `ctrl[n-1]` ∈ [0, 1] is the optimization parameter
- `t[n-1]` and `t[n+1]` are the neighboring time points
- When `ctrl[n-1] = 0`: point n moves to the left neighbor
- When `ctrl[n-1] = 1`: point n moves to the right neighbor

**Number of optimization parameters**: N-2 (one for each interior point)

**Initial guess** for `ctrl[n-1]`:
```
ctrl[n-1] = (t[n] - t[n-1]) / (t[n+1] - t[n-1])
```

This places each point at its current location initially.

## CRITICAL: What NOT to Touch

**This implementation should ONLY add time adjustment functionality. DO NOT modify any existing features.**

### ❌ DO NOT MODIFY:

1. **Existing optimization logic**:
   - Do NOT change `n_points_initial_f` or `n_points_initial_s` functionality
   - Do NOT change how refinement_base is calculated
   - Do NOT change the iteration loop formula: `n_points = round(1 + (n_points_initial - 1) * base^(k-1))`
   - Do NOT change how control grids are generated

2. **Existing parameters in OptimizationParameters**:
   - Do NOT remove or rename any existing fields
   - Do NOT change any default values except to add new ones
   - Existing parameters: `max_evaluations`, `control_times`, `initial_guess_f`, `algorithm`,
     `ftol_rel`, `ftol_abs`, `xtol_rel`, `xtol_abs`, `n_points_final_f`, `n_points_initial_f`,
     `initial_guess_s`, `n_points_final_s`, `n_points_initial_s`, `bounds_f`, `bounds_s`

3. **Print statements and output**:
   - Do NOT remove or change existing print statements for 's' variables
   - Do NOT change iteration progress reporting
   - ONLY add new print statements for the time adjustment phase

4. **Function signatures**:
   - Do NOT remove parameters from `optimize_with_iterative_refinement()`
   - ONLY add the new `optimize_time_points` parameter (with default=False)
   - Do NOT change parameter order

5. **Dual optimization (f and s)**:
   - Do NOT break existing dual optimization functionality
   - Time adjustment must work for both single-variable (f only) and dual (f and s) modes

### ✅ ONLY ADD:

1. New method: `UtilityOptimizer.optimize_time_adjustment()`
2. New parameter: `optimize_time_points: bool = False` to OptimizationParameters
3. New parameter: `optimize_time_points=False` to `optimize_with_iterative_refinement()`
4. Logic in `optimize_with_iterative_refinement()` to call time adjustment after standard iterations
5. Print statements for the time adjustment phase only

### 🧪 Testing Requirements:

1. With `optimize_time_points=False`: Everything must work exactly as before
2. With `optimize_time_points=True`: Time adjustment phase runs after standard iterations
3. All existing test configurations must continue to work unchanged
4. Dual optimization (f and s) must show both f and s printouts in all iterations

## Implementation Steps

### Phase 1: Create Time Adjustment Optimizer Function

**File**: `optimization.py`

**New function**: `optimize_time_adjustment()`

```python
def optimize_time_adjustment(self, initial_control_points, max_evaluations,
                             algorithm=None, ftol_rel=None, ftol_abs=None,
                             xtol_rel=None, xtol_abs=None):
    """
    Optimize the timing of control points while keeping control values fixed.

    Parameters
    ----------
    initial_control_points : list of tuples
        List of (time, value) tuples from previous optimization.
        First and last points remain fixed.
    max_evaluations : int
        Maximum number of objective function evaluations
    algorithm : str, optional
        NLopt algorithm name (default: 'LN_SBPLX')
    ftol_rel, ftol_abs, xtol_rel, xtol_abs : float, optional
        Tolerance parameters

    Returns
    -------
    dict
        Optimization results with adjusted control points
    """
```

**Algorithm**:
1. Extract times and values from `initial_control_points`
2. Calculate initial guess for ctrl parameters
3. Define objective function that:
   - Takes ctrl parameters
   - Reconstructs time points from ctrl
   - Creates control function with (adjusted_times, original_values)
   - Runs model integration
   - Returns discounted utility
4. Run NLopt optimization with bounds [0, 1] on all ctrl parameters
5. Return optimized time points with original values

### Phase 2: Extend Iterative Refinement to Include Time Adjustment

**File**: `optimization.py`

**Modify**: `optimize_with_iterative_refinement()`

**New parameter**: `optimize_time_points` (bool, optional, default=False)

When `optimize_time_points=True`:
- After all standard iterations complete
- Call `optimize_time_adjustment()` with final control points
- Add results to iteration history
- Return combined results

**Implementation location**: Add after line 844 (after final iteration completes)

```python
# After standard iterations
if optimize_time_points:
    print(f"\n{'=' * 80}")
    print(f"  TIME ADJUSTMENT OPTIMIZATION")
    print(f"  Optimizing temporal placement of {n_points_f} control points")
    print(f"  Keeping control values fixed")
    print(f"{'=' * 80}\n")

    time_opt_result = self.optimize_time_adjustment(
        final_result['control_points'],
        max_evaluations,
        algorithm=algorithm,
        ftol_rel=ftol_rel,
        ftol_abs=ftol_abs,
        xtol_rel=xtol_rel,
        xtol_abs=xtol_abs
    )

    # Update final results
    final_result = time_opt_result
    iteration_history.append(time_opt_result)
    total_evaluations += time_opt_result['n_evaluations']
```

### Phase 3: Configuration Support

**File**: `parameters.py`

**Modify**: `OptimizationParameters` dataclass

**New field**: `optimize_time_points` (bool, optional, default=False)

```python
@dataclass
class OptimizationParameters:
    # ... existing fields ...
    optimize_time_points: bool = False  # Enable time adjustment phase
```

**File**: `test_optimization.py`

**Modify**: Line 498-509 (optimize_with_iterative_refinement call)

Add new parameter from config:
```python
opt_results = optimizer.optimize_with_iterative_refinement(
    n_iterations=opt_params.control_times,
    initial_guess_scalar=opt_params.initial_guess_f,
    max_evaluations=max_evaluations,
    algorithm=opt_params.algorithm,
    ftol_rel=opt_params.ftol_rel,
    ftol_abs=opt_params.ftol_abs,
    xtol_rel=opt_params.xtol_rel,
    xtol_abs=opt_params.xtol_abs,
    n_points_final=opt_params.n_points_final_f,
    initial_guess_s_scalar=opt_params.initial_guess_s,
    n_points_final_s=opt_params.n_points_final_s,
    optimize_time_points=opt_params.optimize_time_points  # NEW
)
```

### Phase 4: Dual Optimization (f and s) Support

**Extension**: If optimizing both f and s, adjust times for both control trajectories

**Approach**: Independent time adjustments for f and s
- Separate ctrl parameters for f time points and s time points
- Each trajectory can have different temporal spacing
- Supports different numbers of control points (already implemented)
- If N_f control points for f: optimize N_f - 2 time parameters
- If N_s control points for s: optimize N_s - 2 time parameters
- Total optimization dimension: (N_f - 2) + (N_s - 2)

**Implementation**:
```python
def optimize_time_adjustment(self, initial_f_control_points,
                             initial_s_control_points=None,
                             max_evaluations, algorithm=None, ...):
    """
    Optimize timing of f control points, and optionally s control points.

    Parameters
    ----------
    initial_f_control_points : list of tuples
        (time, f_value) tuples for f trajectory
    initial_s_control_points : list of tuples, optional
        (time, s_value) tuples for s trajectory
        If None, only optimizes f times
    """
```

If `initial_s_control_points` is provided:
- Create combined parameter vector: [f_ctrl_params, s_ctrl_params]
- Objective function reconstructs both time arrays
- Both trajectories optimized simultaneously but independently

### Phase 5: Testing and Validation

**Test cases**:

1. **Single variable (f only)**:
   - Run standard iterative refinement (4 iterations)
   - Add time adjustment optimization
   - Verify time points move to improve objective
   - Check that first and last points remain fixed

2. **Verify improvement**:
   - Objective after time adjustment ≥ objective before
   - Should see modest improvement (timing is secondary to values)

3. **Edge cases**:
   - N=2 control points: no interior points, should skip optimization
   - N=3: single interior point, 1D optimization
   - Large N: verify convergence with many parameters

4. **Dual optimization**:
   - Test with both f and s optimized
   - Verify both trajectories adjusted correctly

### Phase 6: Documentation

**Update files**:

1. **README.md**: Add section describing time adjustment optimization
2. **TIME_ADJUSTMENT_IMPLEMENTATION_PLAN.md** (this file): Keep as design reference
3. **Docstrings**: Complete documentation in `optimize_time_adjustment()`

## Configuration Example

**File**: `config_test_time_adjust.json`

```json
{
  "run_name": "test_time_adjustment",
  "optimization_parameters": {
    "control_times": 4,
    "initial_guess_f": 0.5,
    "max_evaluations": 5000,
    "n_points_final_f": 10,
    "algorithm": "LN_SBPLX",
    "xtol_abs": 1e-10,
    "optimize_time_points": true
  }
}
```

This would:
1. Run 4 iterations of standard refinement
2. End with 10 control points
3. Run time adjustment on those 10 points (optimizing 8 interior times)
4. Return final control trajectory with adjusted timing

## Expected Benefits

1. **Better temporal resolution**: Control points concentrate where changes matter most
2. **Improved objective**: Fine-tuning timing can provide modest gains
3. **Physical insight**: Shows when interventions should occur vs. what they should be
4. **Robustness**: Separates value optimization from timing optimization

## Potential Challenges

1. **Constraint handling**: Ensure time points remain ordered (t[n-1] < t[n] < t[n+1])
   - Current parameterization automatically enforces this via ctrl ∈ [0,1]

2. **Local minima**: Time adjustment may have multiple local optima
   - Mitigated by good initial guess (current times)

3. **Computation cost**: Additional optimization phase adds evaluations
   - Worth it if improvement is significant
   - Can be disabled via configuration flag

4. **Dual optimization complexity**: Coordinating f and s time adjustments
   - Using independent time adjustments for maximum flexibility
   - More parameters to optimize but better final result

## Success Criteria

1. ✅ Implementation compiles and runs without errors
2. ✅ Time points move to improve objective (or stay same if already optimal)
3. ✅ First and last time points remain fixed
4. ✅ Interior time points remain ordered
5. ✅ Backward compatible (disabled by default)
6. ✅ Works with both single (f) and dual (f+s) optimization
7. ✅ Documentation complete

## Timeline Estimate

- Phase 1: Core optimizer function - **2 hours**
- Phase 2: Integration with iterative refinement - **1 hour**
- Phase 3: Configuration support - **30 minutes**
- Phase 4: Dual optimization support - **1.5 hours** (independent f and s)
- Phase 5: Testing - **1.5 hours**
- Phase 6: Documentation - **30 minutes**

**Total: ~7 hours** of development time

## Files to Modify

1. `optimization.py`: Add `optimize_time_adjustment()` method
2. `optimization.py`: Modify `optimize_with_iterative_refinement()`
3. `parameters.py`: Add `optimize_time_points` field
4. `test_optimization.py`: Pass new parameter to optimizer
5. `README.md`: Document new feature
6. **New**: `TIME_ADJUSTMENT_IMPLEMENTATION_PLAN.md` (this file)

## Design Decisions

1. **Should time adjustment also apply to s control points?**
   - ✅ **Decision**: Yes, with independent time adjustments for f and s

2. **Should we allow multiple time adjustment iterations?**
   - ✅ **Decision**: Single optimization pass only - one time adjustment after standard refinement completes

3. **Should time adjustment use different tolerance settings?**
   - ✅ **Decision**: Use same tolerances as main optimization

4. **What if objective gets worse after time adjustment?**
   - ✅ **Decision**: Won't happen - starting from current positions (good initial guess) ensures we find equal or better solution
   - No need for fallback logic
