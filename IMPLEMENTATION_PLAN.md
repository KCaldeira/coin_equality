# Implementation Plan: Option Switches for Policy Scenarios

## Overview

This document outlines planned code modifications to enable flexible configuration of damage, taxation, and redistribution policies. The goal is to separate orthogonal policy choices into independent switches that can be combined to explore different scenarios.

---

## 1. Damage Function Separation

Currently, the damage function combines aggregate damage calculation with its distribution across income levels. We will separate these into two independent components.

### 1.1 Aggregate Damage Calculation

**Switch: `income_dependent_aggregate_damage`** (boolean)

| Value | Description |
|-------|-------------|
| `false` | Fractional damage is independent of income level (as in DICE). Damage fraction depends only on temperature. |
| `true` | Total fractional damage decreases as the world gets richer. Wealthier societies can better adapt/protect themselves. |

### 1.2 Damage Distribution

**Switch: `income_dependent_damage_distribution`** (boolean)

| Value | Description |
|-------|-------------|
| `false` | Climate damage is distributed uniformly across the income distribution (same fractional loss for all income levels). |
| `true` | Damage is weighted towards people with low income (poor suffer disproportionately from climate impacts). Uses `y_damage_distribution_halfsat` parameter. |

### 1.3 Parameter Rename

- `y_damage_halfsat` → `y_damage_distribution_halfsat` (clarifies this controls distribution, not aggregate damage)

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

    "income_dependent_damage_distribution": true,
    "_income_dependent_damage_distribution": "If true, damage weighted towards low-income (uses y_damage_distribution_halfsat)",

    "y_damage_distribution_halfsat": 10000.0,
    "_y_damage_distribution_halfsat": "Income level ($) at which climate damage is half of maximum (only used when income_dependent_damage_distribution=true)",

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

## Questions to Resolve

- [ ] Should `uniform_dividend` include the people paying taxes, or only non-payers?
- [ ] For `tax_richest`, what threshold defines "richest"? Top percentile? Above median?
- [ ] For `targeted_lowest_income`, what threshold defines eligibility?
- [ ] How do these policies interact with the existing Gini dynamics?
