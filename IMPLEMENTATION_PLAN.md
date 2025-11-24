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

**Switch: `tax_policy_type`**

| Option | Description | Priority |
|--------|-------------|----------|
| `uniform_fractional` | Uniform fractional tax independent of income (effectively what is in DICE now). Everyone pays the same fraction of their income. | High |
| `tax_richest` | Tax only the richest (aggregate utility optimizing). Concentrates burden on high-income individuals to maximize total utility. | High |
| `uniform_utility_reduction` | Tax designed so everyone experiences the same utility reduction. A "fair" tax system where the burden feels equal. | Lower |

---

## 3. Redistribution Policies

How revenues or benefits from climate policy are distributed back to the population.

**Switch: `redistribution_policy_type`**

| Option | Description |
|--------|-------------|
| `uniform_dividend` | Everyone gets the same per-capita cash dividend (including those who contributed). Universal basic dividend approach. |
| `targeted_lowest_income` | Benefits go only to those with lowest income (aggregate utility optimizing). Maximizes utility gain from redistribution. |

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

    "tax_policy_type": "uniform_fractional",
    "_tax_policy_type": "Options: uniform_fractional, tax_richest, uniform_utility_reduction",

    "redistribution_policy_type": "uniform_dividend",
    "_redistribution_policy_type": "Options: uniform_dividend, targeted_lowest_income"
}
```

### Files to Modify

1. **economic_model.py**: Core logic for damage, tax, and redistribution calculations
2. **utility_functions.py**: Utility calculations for different policy scenarios
3. **parameters.py**: Add new parameter handling
4. **output.py**: Ensure new variables are captured in output
5. **README.md**: Document new configuration options

### Testing Strategy

Create config files that test each combination:
- DICE baseline (all defaults)
- Progressive damage + regressive tax
- Regressive damage + targeted redistribution
- etc.

---

## Priority Order

0. **Phase 0**: Add new configuration keywords to parameter loading code (parameters.py)
1. **Phase 1**: Separate aggregate damage from damage distribution
2. **Phase 2**: Implement tax policy options (uniform_fractional, tax_richest)
3. **Phase 3**: Implement redistribution policy options
4. **Phase 4**: Add uniform_utility_reduction tax option (lower priority)

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

Pass the key parameters that characterize the income distribution at the end of the previous time step:

| Quantity | Description |
|----------|-------------|
| `y_mean_prev` | Mean income from previous time step |
| `gini_prev` | Gini coefficient from previous time step |

**Important:** Always pass `gini`, never `a`. The Pareto parameter `a` is a derived quantity under the assumption of a Pareto-Lorenz distribution. The Gini coefficient is the fundamental state variable. Functions that need `a` should compute it internally using `a_from_G(gini)`.

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
def calculate_tendencies(state, params, prev_income_dist):
    """
    Parameters
    ----------
    state : dict
        Current state variables
    params : dict
        Model parameters (static)
    prev_income_dist : dict
        Income distribution from previous time step:
        {'y_mean': float, 'gini': float}
    """
```

*Pros:* Clear separation of concerns, explicit dependency
*Cons:* Requires updating all callers

**Recommended: Option B** - The explicit argument makes the temporal dependency clear and separates static parameters from time-varying state information.

### Initialization

For the first time step (t=0), use the initial income distribution from the state variables:

```python
prev_income_dist = {
    'y_mean': state['y_mean'][0],
    'gini': state['gini'][0],
}
```

### Usage in Damage Calculations

When `income_dependent_damage_distribution=true`:

```python
# Use previous time step's income distribution for damage weighting
gini = prev_income_dist['gini']
y_mean = prev_income_dist['y_mean']

# Calculate damage distribution using previous income structure
# (functions compute 'a' internally from gini)
damage_weight = pareto_damage_weight(y, gini, y_mean, y_damage_distribution_scale)
```

Similarly for taxation and redistribution calculations that depend on the income distribution.

### Refactoring: Functions to Change from `a` to `gini`

The following functions currently take the Pareto parameter `a` as an argument. They should be refactored to take `gini` instead and compute `a` internally using `a_from_G(gini)`:

| File | Function | Current Signature | New Signature |
|------|----------|-------------------|---------------|
| `climate_damage_distribution.py` | `pareto_integral_scipy` | `(c_mean, a, c_scale)` | `(c_mean, gini, c_scale)` |

**Note:** The unit test files (`unit_test_eq1.2.py`, etc.) may keep `a` in their function signatures since they are testing specific mathematical formulas that are naturally expressed in terms of `a`. The conversion functions `a_from_G()` and `G_from_a()` in `income_distribution.py` are utilities and should remain as-is.

**Good patterns already in codebase (take `gini`, compute `a` internally):**
- `L_pareto(F, G)` in `income_distribution.py`
- `L_pareto_derivative(F, G)` in `income_distribution.py`

### Consistency Check

For validation, we can optionally compute the "error" between:
- Income distribution used for damage calculation (from previous step)
- Income distribution resulting from damage calculation (current step)

This error should be small if the time step is appropriate. Large errors might indicate the need for smaller time steps.

---

## Questions to Resolve

- [ ] Should `uniform_dividend` include the people paying taxes, or only non-payers?
- [ ] For `tax_richest`, what threshold defines "richest"? Top percentile? Above median?
- [ ] For `targeted_lowest_income`, what threshold defines eligibility?
- [ ] How do these policies interact with the existing Gini dynamics?

## Resolved Questions

- [x] **Should the Pareto parameter `a` be passed explicitly, or always derived from `gini`?**
  - **Decision:** Always pass `gini`, never `a`. The Gini coefficient is the fundamental state variable; `a` is a derived quantity under the Pareto-Lorenz assumption. Functions that need `a` should compute it internally using `a_from_G(gini)`.
