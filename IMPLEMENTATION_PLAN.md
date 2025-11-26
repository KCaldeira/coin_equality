# Implementation Plan: Option Switches for Policy Scenarios

## Overview

This document outlines planned code modifications to enable flexible configuration of damage, taxation, and redistribution policies. The goal is to separate orthogonal policy choices into independent switches that can be combined to explore different scenarios.

---

## 1. Damage Function Separation

Currently, the damage function combines aggregate damage calculation with its distribution across income levels. We will separate these into two independent components.

### 1.1 Variable Rename

- `omega_max` → `omega_aggregate` (clarifies this is the aggregate damage fraction)
- `y_damage_halfsat` → `y_damage_distribution_scale` (clarifies this controls distribution, not aggregate damage)

### 1.2 Aggregate Damage Calculation

**Switch: `income_dependent_aggregate_damage`** (boolean)

| Value | Description |
|-------|-------------|
| `false` | Fractional damage is independent of income level (as in DICE). Damage fraction depends only on temperature. |
| `true` | Total fractional damage decreases as the world gets richer. Wealthier societies can better adapt/protect themselves. |

**Implementation:**

```python
# Base damage from temperature (always calculated)
omega_base = psi1 * delta_T + psi2 * delta_T**2

if income_dependent_aggregate_damage:
    # Wealthier societies experience less aggregate damage
    income_damage_scale = pareto_integral_scipy(y_mean, a, y_damage_aggregate_halfsat)
    omega_aggregate = omega_base * income_damage_scale
else:
    # DICE-like: damage independent of income
    omega_aggregate = omega_base
```

**New function to add:**

```python
import numpy as np
from scipy.special import gammaincc, gamma

def pareto_integral_scipy(c_mean, a, c_scale):
    """
    Computes the income-dependent damage scaling factor.

    This integral represents how aggregate damage scales with mean income
    for a Pareto-distributed population. As c_mean increases relative to
    c_scale, the integral decreases, representing better adaptation capacity.

    Parameters
    ----------
    c_mean : float
        Mean income (y_mean in model).
    a : float
        Pareto parameter (>1), derived from Gini coefficient.
    c_scale : float
        Income scale for damage saturation (y_damage_aggregate_halfsat).

    Returns
    -------
    float
        Scaling factor for aggregate damage (0 to 1, decreasing with wealth).
    """
    # k = c_mean * (1 - 1/a)
    k = c_mean * (1.0 - 1.0 / a)

    # s = 1 - a
    s = 1.0 - a

    # Argument of the incomplete gamma
    x = k / c_scale

    # Upper incomplete gamma Γ(s, x)
    # SciPy: Γ(s, x) = gammaincc(s, x) * gamma(s)
    gamma_upper = gammaincc(s, x) * gamma(s)

    # Full analytic expression
    result = a * (k ** a) * (c_scale ** (1.0 - a)) * gamma_upper

    return result
```

**New parameter needed:**

- `y_damage_aggregate_halfsat`: Income scale for aggregate damage saturation ($). Controls how quickly aggregate damage decreases as society gets wealthier. Only used when `income_dependent_aggregate_damage=true`.

### 1.3 Damage Distribution

**Switch: `income_dependent_damage_distribution`** (boolean)

| Value | Description |
|-------|-------------|
| `false` | Climate damage is distributed uniformly across the income distribution (same fractional loss for all income levels). |
| `true` | Damage is weighted towards people with low income (poor suffer disproportionately from climate impacts). Uses `y_damage_distribution_scale` parameter. |

**Implementation:**

When `income_dependent_damage_distribution=false`:
- All income levels experience the same fractional damage (`omega_aggregate`)
- No change to Gini coefficient from climate damage (`Gini_climate = Gini`)

When `income_dependent_damage_distribution=true`:
- Use current regressive damage formula with `y_damage_distribution_scale`
- Climate damage increases Gini (poor lose more)

### 1.4 Four Possible Combinations

| `income_dependent_aggregate_damage` | `income_dependent_damage_distribution` | Behavior |
|-------------------------------------|----------------------------------------|----------|
| `false` | `false` | DICE-like: uniform damage, no income effects |
| `false` | `true` | Fixed aggregate damage, but poor suffer more (current default) |
| `true` | `false` | Aggregate damage decreases with wealth, but distributed uniformly |
| `true` | `true` | Both effects: wealthy societies have less damage AND poor suffer more |

---

## 2. Tax/Abatement Cost Policies

How the costs of climate policy (abatement) are distributed across the population.

**Switch: `income_dependent_tax_policy`** (boolean)

| Value | Description |
|-------|-------------|
| `false` | Uniform fractional tax independent of income (effectively what is in DICE now). Everyone pays the same fraction of their income. |
| `true` | Progressive tax - tax only the richest (aggregate utility optimizing). Concentrates burden on high-income individuals to maximize total utility. |

---

## 3. Redistribution Policies

How revenues or benefits from climate policy are distributed back to the population.

### 3.1 Redistribution Enable/Disable

**Switch: `income_redistribution`** (boolean)

| Value | Description |
|-------|-------------|
| `false` | No redistribution - all revenues from carbon pricing go to abatement only |
| `true` | Enable redistribution - revenues can be redistributed according to policy |

### 3.2 Redistribution Distribution Policy

**Switch: `income_dependent_redistribution_policy`** (boolean)

*Only applies when `income_redistribution=true`*

| Value | Description |
|-------|-------------|
| `false` | Uniform dividend - everyone gets the same per-capita cash dividend (including those who contributed). Universal basic dividend approach. |
| `true` | Targeted redistribution - benefits go only to those with lowest income (aggregate utility optimizing). Maximizes utility gain from redistribution. |

---

## Implementation Notes

### Configuration Structure

These switches should be added to the `scalar_parameters` section of config files:

```json
"scalar_parameters": {
    "income_dependent_aggregate_damage": false,
    "_income_dependent_aggregate_damage": "If true, aggregate damage decreases as world gets richer",

    "y_damage_aggregate_halfsat": 10000.0,
    "_y_damage_aggregate_halfsat": "Income scale ($) for aggregate damage saturation (only used when income_dependent_aggregate_damage=true)",

    "income_dependent_damage_distribution": true,
    "_income_dependent_damage_distribution": "If true, damage weighted towards low-income (uses y_damage_distribution_scale)",

    "y_damage_distribution_scale": 10000.0,
    "_y_damage_distribution_scale": "Income level ($) at which climate damage is half of maximum (only used when income_dependent_damage_distribution=true)",

    "income_dependent_tax_policy": false,
    "_income_dependent_tax_policy": "If true, progressive tax (tax richest); if false, uniform fractional tax",

    "income_redistribution": true,
    "_income_redistribution": "If true, enable redistribution; if false, no redistribution (all revenues to abatement)",

    "income_dependent_redistribution_policy": false,
    "_income_dependent_redistribution_policy": "If true, targeted to lowest income; if false, uniform dividend (only applies when income_redistribution=true)",

    "n_discrete": 1000,
    "_n_discrete": "Number of discrete segments for numerical integration over income distribution (integer)"
}
```

### Files to Modify

1. **economic_model.py**: Core logic for damage, tax, and redistribution calculations
2. **policy_functions.py**: New module with utility functions for policy-dependent calculations (Section 5)
3. **climate_damage_distribution.py**: Update to use previous_step_values with critical ranks
4. **parameters.py**: Add new parameter handling
5. **output.py**: Ensure new variables are captured in output (including F_crit values)
6. **README.md**: Document new configuration options

### Testing Strategy

Create config files that test each combination:
- DICE baseline (all defaults)
- Progressive damage + regressive tax
- Regressive damage + targeted redistribution
- etc.

---

## 5. Utility Functions for Policy-Dependent Calculations

To support the various `income_dependent_*` policy switches, we need utility functions that compute income, damage, and utility distributions accounting for taxation, redistribution, and climate damage.

### 5.1 Core Utility Functions

These functions must account for all policy switches and provide the building blocks for economic calculations:

#### Function 1: `income_at_rank(F, base_income_dist, damage_params, tax_params, redistribution_params)`

**Purpose**: Compute income at rank F after applying climate damage, taxation, and redistribution.

**Parameters**:
- `F`: Rank in the distribution (0 to 1)
- `base_income_dist`: Dictionary with `{'y_mean', 'gini'}`
- `damage_params`: Climate damage parameters
- `tax_params`: Taxation policy parameters
- `redistribution_params`: Redistribution policy parameters

**Returns**: Income at rank F after all transformations

**Logic**:
1. Start with base Pareto income at rank F: `c(F) = y_mean * (1 - 1/a) * (1 - F)^(-1/a)`
2. Apply climate damage based on income level (if `income_dependent_damage_distribution=true`)
3. Apply taxation:
   - If `income_dependent_tax_policy=false`: uniform fractional tax
   - If `income_dependent_tax_policy=true`: progressive tax (tax only above F_crit_tax)
4. Apply redistribution:
   - If `income_dependent_redistribution_policy=false`: uniform dividend to all
   - If `income_dependent_redistribution_policy=true`: targeted to F < F_crit_redistribution

#### Function 2: `integrated_damage_scaling(base_income_dist, damage_params, tax_params, redistribution_params)`

**Purpose**: Compute aggregate damage as fraction of GDP, integrating over the entire distribution.

**Returns**: Float, aggregate damage scaling factor (0 to 1)

**Logic**:
```
∫₀¹ damage_at_rank(F) * income_at_rank(F) dF / ∫₀¹ income_at_rank(F) dF
```

Where `damage_at_rank(F)` depends on:
- Base temperature-dependent damage (ψ₁·ΔT + ψ₂·ΔT²)
- Income-dependent aggregate scaling (if `income_dependent_aggregate_damage=true`)
- Income-dependent damage distribution (if `income_dependent_damage_distribution=true`)
- Redistribution and taxation effects on the income distribution

#### Function 3: `integrated_utility(base_income_dist, damage_params, tax_params, redistribution_params, eta)`

**Purpose**: Compute aggregate utility integrating over the post-damage, post-tax, post-redistribution income distribution.

**Returns**: Float, mean utility

**Logic**:
```
∫₀¹ u(income_at_rank(F)) dF
```

Where `u(c) = c^(1-η)/(1-η)` for η≠1, or `u(c) = log(c)` for η=1.

### 5.2 Temporal Structure

These functions are called with different temporal contexts:

| Calculation | Uses Distribution From | Uses Decisions From | Purpose |
|-------------|----------------------|-------------------|---------|
| **Climate Damage** | Previous time step (t-1) | Previous time step (t-1) | Calculate Ω based on population vulnerability at start of period |
| **Taxation & Redistribution** | Current time step (t) | Current time step (t) | Apply current policy decisions to current distribution |
| **Utility** | Current time step (t) | Current time step (t) | Evaluate welfare under current conditions |

**Workflow in `calculate_tendencies()`**:

1. **Damage calculation** (using previous time step):
   ```python
   Omega = integrated_damage_scaling(
       previous_step_values,  # t-1 distribution
       damage_params,
       prev_tax_params,   # t-1 tax decisions
       prev_redistribution_params  # t-1 redistribution decisions
   )
   ```

2. **Current step calculations** (using current time step):
   ```python
   # Apply damage to get current damaged income distribution
   Y_damaged = Y_gross * (1 - Omega)
   current_damaged_dist = compute_damaged_distribution(Y_damaged, Gini_climate)

   # Calculate utility with current policies
   U = integrated_utility(
       current_damaged_dist,  # t distribution (post-damage)
       damage_params,
       current_tax_params,    # t tax decisions
       current_redistribution_params  # t redistribution decisions
   )
   ```

### 5.3 Critical Rank Calculation

For progressive taxation and targeted redistribution, we need to determine critical ranks:

#### `calculate_F_crit_tax(y_mean, gini, fract_gdp, damage_params)`

Finds the rank F_crit above which individuals are taxed (progressive tax).

**Constraint**: Total tax revenue = `fract_gdp * f * Y_damaged`

#### `calculate_F_crit_redistribution(y_mean, gini, fract_gdp, f, redistribution_budget)`

Finds the rank F_crit below which individuals receive redistribution (targeted policy).

**Constraint**: Total redistribution = `fract_gdp * (1-f) * Y_damaged`

### 5.4 Implementation Location

Create these functions in a new module: **`policy_functions.py`**

This module will be called by:
- `economic_model.py`: For calculating damage, utility, and redistribution effects
- `optimization.py`: For evaluating objective function

### 5.5 Analytical vs. Numerical Integration

- **Preferred approach**: Analytical solutions using hypergeometric functions (following existing pattern)
- **Fallback**: Numerical integration for complex cases
  - Use discrete approximation: divide rank interval [0,1] into `n_discrete` segments
  - Parameter `n_discrete` (default: 1000) controls accuracy vs. performance tradeoff
  - For each segment i: F_i = i/n_discrete, width = 1/n_discrete
  - Approximate integral: Σ f(F_i) * (1/n_discrete)
- **Alternative**: `scipy.integrate.quad` for adaptive quadrature when needed
- **Documentation**: Each function should document whether it uses analytical or numerical methods

---

## Priority Order

0. **Phase 0**: Add new configuration keywords to parameter loading code (parameters.py)
1. **Phase 1**: Separate aggregate damage from damage distribution
2. **Phase 2**: Implement temporal income distribution approach (Option B) - **COMPLETED**
3. **Phase 3**: Implement boolean policy switches - **COMPLETED**
4. **Phase 4**: Create utility functions in `policy_functions.py` (Section 5)
   - `income_at_rank()`: Income after damage/tax/redistribution at rank F
   - `integrated_damage_scaling()`: Aggregate damage accounting for policies
   - `integrated_utility()`: Mean utility accounting for policies
   - `calculate_F_crit_tax()`: Critical rank for progressive taxation
   - `calculate_F_crit_redistribution()`: Critical rank for targeted redistribution
5. **Phase 5**: Implement tax policy (uniform vs. progressive)
6. **Phase 6**: Implement redistribution policy (uniform dividend vs. targeted)
7. **Phase 7**: Testing and validation across all policy combinations

---

## 4. Temporal Income Distribution for Damage Calculations

### The Problem

Within `calculate_tendencies()`, computing climate damage requires knowledge of the income distribution. However, the income distribution itself depends on:
- Climate damage (which reduces income, especially for the poor if `income_dependent_damage_distribution=true`)
- Taxation policy (which removes income from certain groups)
- Redistribution policy (which adds income to certain groups)

This creates a circular dependency: we need the income distribution to calculate damage, but damage affects the income distribution. One approach would be to iterate within `calculate_tendencies()` until reaching an internally consistent solution, but this is computationally expensive and adds complexity.

### The Solution: Use Previous Time Step's Income Distribution

Instead of iterating to internal consistency, we use the income distribution from the **end of the previous time step** to estimate climate damage on the **current time step**. This is physically reasonable because:

1. Climate damage in a given year depends on the vulnerability of the population at the start of that year
2. The time step is small enough that the income distribution changes gradually
3. This approach is numerically stable and avoids iteration

### Information to Pass

**Preferred approach: Key quantities** (rather than passing a continuous function)

Pass the key parameters that characterize the income distribution and policy decisions at the end of the previous time step:

| Quantity | Description |
|----------|-------------|
| `y_mean` | Mean income from previous time step |
| `gini` | Gini coefficient from previous time step |
| `F_crit_tax` | Critical rank for taxation from previous time step (only if `income_dependent_tax_policy=true`) |
| `F_crit_redistribution` | Critical rank for redistribution from previous time step (only if `income_dependent_redistribution_policy=true`) |

**Important:** Always pass `gini`, never `a`. The Pareto parameter `a` is a derived quantity under the assumption of a Pareto-Lorenz distribution. The Gini coefficient is the fundamental state variable. Functions that need `a` should compute it internally using `a_from_G(gini)`.

**Rationale for including critical ranks:**
- Climate damage depends on the income distribution **and** policy decisions from the previous time step
- Progressive taxation and targeted redistribution affect the effective income distribution
- To correctly calculate damage, we need to know where tax/redistribution boundaries were in the previous period

These quantities fully characterize a Pareto distribution, which is sufficient for all damage, taxation, and redistribution calculations.

**Why key quantities over continuous function:**
- More numerically accurate (no interpolation errors)
- Simpler to implement and debug
- Directly usable in analytical formulas (e.g., `pareto_integral_scipy`)
- The Pareto assumption is already embedded in our model

### Implementation Options

**Option A: Add to `params` dictionary**

Pass the previous income distribution as new fields in `params`:

```python
params = {
    # ... existing parameters ...
    'y_mean_prev': y_mean_prev,
    'gini_prev': gini_prev,
}
```

*Pros:* Minimal API change, consistent with existing pattern
*Cons:* Mixes time-varying state with static parameters

**Option B: Add new argument to `calculate_tendencies()`**

```python
def calculate_tendencies(state, params, previous_step_values):
    """
    Parameters
    ----------
    state : dict
        Current state variables
    params : dict
        Model parameters (static)
    previous_step_values : dict
        Income distribution and policy decisions from previous time step:
        {
            'y_mean': float,
            'gini': float,
            'F_crit_tax': float (optional, only if income_dependent_tax_policy=true),
            'F_crit_redistribution': float (optional, only if income_dependent_redistribution_policy=true)
        }
    """
```

*Pros:* Clear separation of concerns, explicit dependency
*Cons:* Requires updating all callers

**Recommended: Option B** - The explicit argument makes the temporal dependency clear and separates static parameters from time-varying state information.

**Status: IMPLEMENTED** - Option B has been implemented in `economic_model.py`:
- `calculate_tendencies()` now takes `previous_step_values` as a required argument
- `calculate_tendencies()` returns `current_income_dist` in its results dictionary (needs update to include F_crit values)
- `integrate_model()` initializes `previous_step_values` from initial conditions and updates it each time step

### Initialization

For the first time step (t=0), use initial gross income and background Gini:

```python
# In integrate_model():
Y_gross_initial = A0 * (K0 ** alpha) * (L0 ** (1 - alpha))
y_gross_initial = Y_gross_initial / L0
Gini_initial = config.time_functions['Gini_background'](t_start)

previous_step_values = {
    'y_mean': y_gross_initial,
    'gini': Gini_initial,
}

# Initialize critical ranks if needed
if params['income_dependent_tax_policy']:
    previous_step_values['F_crit_tax'] = calculate_F_crit_tax(
        y_gross_initial, Gini_initial, params['fract_gdp'], params
    )

if params['income_dependent_redistribution_policy']:
    previous_step_values['F_crit_redistribution'] = calculate_F_crit_redistribution(
        y_gross_initial, Gini_initial, params['fract_gdp'], params['f'], params
    )
```

### Usage in Damage Calculations

Climate damage is calculated using the previous time step's income distribution **and** policy decisions:

```python
# Calculate damage using previous time step's complete state
Omega, Gini_climate = calculate_climate_damage_from_prev_distribution(
    delta_T, previous_step_values, params
)
```

Inside `calculate_climate_damage_from_prev_distribution()`:

```python
y_mean = previous_step_values['y_mean']
gini = previous_step_values['gini']

# Get critical ranks if policies are income-dependent
if params['income_dependent_tax_policy']:
    F_crit_tax = previous_step_values['F_crit_tax']
else:
    F_crit_tax = 1.0  # No one is taxed

if params['income_dependent_redistribution_policy']:
    F_crit_redistribution = previous_step_values['F_crit_redistribution']
else:
    F_crit_redistribution = 0.0  # No one receives redistribution

# Integrate damage over the distribution, accounting for:
# - Income-dependent damage function
# - Tax-modified income above F_crit_tax
# - Redistribution-modified income below F_crit_redistribution
Omega = integrated_damage_scaling(y_mean, gini, F_crit_tax, F_crit_redistribution, params)
```

### Refactoring: Functions to Change from `a` to `gini`

The following functions currently take the Pareto parameter `a` as an argument. They should be refactored to take `gini` instead and compute `a` internally using `a_from_G(gini)`:

| File | Function | Current Signature | New Signature |
|------|----------|-------------------|---------------|
| `climate_damage_distribution.py` | `pareto_integral_scipy` | `(c_mean, a, c_scale)` | `(c_mean, gini, c_scale)` |

**Note:** The unit test files (`unit_test_eq1.2.py`, etc.) may keep `a` in their function signatures since they are testing specific mathematical formulas that are naturally expressed in terms of `a`. The conversion functions `a_from_G()` and `G_from_a()` in `income_distribution.py` are utilities and should remain as-is.

**Good patterns already in codebase (take `gini`, compute `a` internally):**
- `L_pareto(F, G)` in `income_distribution.py`
- `L_pareto_derivative(F, G)` in `income_distribution.py`

### Return Value from `calculate_tendencies()`

`calculate_tendencies()` must return `current_income_dist` with all necessary information for the next time step:

```python
# At end of calculate_tendencies():
current_income_dist = {
    'y_mean': y_net,  # Net per-capita income after damage/tax/redistribution
    'gini': G_eff,    # Effective Gini after damage/tax/redistribution
}

# Add critical ranks for next time step
if params['income_dependent_tax_policy']:
    current_income_dist['F_crit_tax'] = calculate_F_crit_tax(
        y_net, G_eff, params['fract_gdp'], params
    )

if params['income_dependent_redistribution_policy']:
    current_income_dist['F_crit_redistribution'] = calculate_F_crit_redistribution(
        y_net, G_eff, params['fract_gdp'], params['f'], params
    )

results['current_income_dist'] = current_income_dist
return results
```

### Consistency Check

For validation, we can optionally compute the "error" between:
- Income distribution used for damage calculation (from previous step)
- Income distribution resulting from damage calculation (current step)

This error should be small if the time step is appropriate. Large errors might indicate the need for smaller time steps.

---

## Questions to Resolve

- [ ] For uniform dividend (`income_dependent_redistribution_policy=false`): Should it include the people paying taxes, or only non-payers?
- [ ] For progressive tax (`income_dependent_tax_policy=true`): What threshold defines "richest"? Top percentile? Above median?
- [ ] For targeted redistribution (`income_dependent_redistribution_policy=true`): What threshold defines eligibility for benefits?
- [ ] How do these policies interact with the existing Gini dynamics?

## Resolved Questions

- [x] **Should the Pareto parameter `a` be passed explicitly, or always derived from `gini`?**
  - **Decision:** Always pass `gini`, never `a`. The Gini coefficient is the fundamental state variable; `a` is a derived quantity under the Pareto-Lorenz assumption. Functions that need `a` should compute it internally using `a_from_G(gini)`.

- [x] **Should policy switches be string enums or booleans?**
  - **Decision:** All policy switches are now booleans with the naming convention `income_dependent_*`. This simplifies the API and makes the policy choices clearer:
    - `income_dependent_tax_policy`: `false` = uniform fractional, `true` = progressive (tax richest)
    - `income_dependent_redistribution_policy`: `false` = uniform dividend, `true` = targeted to lowest income
  - Additional policy options (like `uniform_utility_reduction` tax) can be added later as separate boolean switches if needed.
