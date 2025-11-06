# COIN_equality

A simple-as-possible stylized representation of the tradeoff between investment in income redistribution versus investment in emissions abatement.

## Table of Contents

- [Overview](#overview)
- [Model Structure](#model-structure)
  - [Objective Function](#objective-function)
  - [Calculation Order](#calculation-order)
  - [Core Components](#core-components)
- [Key Parameters](#key-parameters)
  - [Scalar Parameters (Time-Invariant)](#scalar-parameters-time-invariant)
  - [Time-Dependent Functions](#time-dependent-functions)
  - [Control Variables](#control-variables)
  - [Integration Parameters](#integration-parameters)
  - [Initial Conditions (Computed Automatically)](#initial-conditions-computed-automatically)
- [Model Features](#model-features)
  - [Simplifying Assumptions](#simplifying-assumptions)
  - [Key Insights](#key-insights)
- [Implementation: Key Functions](#implementation-key-functions)
- [Parameter Organization](#parameter-organization)
  - [Configuration File Structure](#configuration-file-structure)
  - [Initial Conditions](#initial-conditions)
  - [Example Configuration](#example-configuration)
  - [Loading Configuration](#loading-configuration)
- [Unit Testing: Validating Analytical Solutions](#unit-testing-validating-analytical-solutions)
  - [Unit Test for Equation (1.2): Climate Damage](#unit-test-for-equation-12-climate-damage)
  - [Testing the Forward Model](#testing-the-forward-model)
  - [Testing the Optimization with Parameter Overrides](#testing-the-optimization-with-parameter-overrides)
- [Time Integration](#time-integration)
  - [Integration Function](#integration-function)
  - [Implementation Notes](#implementation-notes)
  - [Performance Optimizations](#performance-optimizations)
  - [Output Variables](#output-variables)
- [Output and Visualization](#output-and-visualization)
  - [Saving Results](#saving-results)
  - [Output Files](#output-files)
  - [Example Workflow](#example-workflow)
- [Optimization Configuration](#optimization-configuration)
  - [Direct Multi-Point Optimization](#direct-multi-point-optimization)
  - [Iterative Refinement Optimization](#iterative-refinement-optimization)
  - [Optimization Stopping Criteria](#optimization-stopping-criteria)
  - [Dual Optimization (f and s)](#dual-optimization-f-and-s)
- [Next Steps](#next-steps)
- [Dual Optimization of Savings Rate and Abatement Allocation](#dual-optimization-of-savings-rate-and-abatement-allocation)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)
- [Authors](#authors)

## Overview

This project develops a highly stylized model of an economy with income inequality, where a specified fraction of gross production is allocated to social good. The central question is how to optimally allocate resources between two competing objectives:

1. **Income redistribution** - reducing inequality by transferring income from high-income to low-income individuals
2. **Emissions abatement** - reducing carbon emissions to mitigate future climate damage

The model extends the COIN framework presented in [Caldeira et al. (2023)](https://doi.org/10.1088/1748-9326/acf949) to incorporate income inequality and diminishing marginal utility of income.

## Model Structure

### Objective Function

The model optimizes the time-integral of aggregate utility by choosing the allocation fraction `f(t)` between emissions abatement and income redistribution:

```
max∫₀^∞ e^(-ρt) · U(t) · L(t) dt,  subject to 0 ≤ f(t) ≤ 1
```

where:
- `ρ` = pure rate of time preference
- `U(t)` = mean utility of the population at time t
- `L(t)` = population at time t
- `f(t)` = fraction of resources allocated to abatement (control variable)

### Calculation Order

For the differential equation solver, variables are calculated in this order:

1. **Y_gross** from K, L, A, α (Eq 1.1: Cobb-Douglas production)
2. **ΔT** from Ecum, k_climate (Eq 2.2: temperature from cumulative emissions)
3. **y_gross** from Y_gross, L (mean per-capita gross income before climate damage)
4. **Ω, G_climate** from ΔT, Gini, y_gross, damage params (Eq 1.2: income-dependent climate damage and distributional effect; when f_gdp >= 1, uses uniform damage approximation)
5. **Y_damaged** from Y_gross, Ω (Eq 1.3: production after climate damage)
6. **y** from Y_damaged, L, s (Eq 1.4: mean per-capita income after climate damage)
7. **c_redist** from y, f_gdp (Eq 4.3: per-capita amount redistributable)
8. **E_pot** from σ, Y_gross (Eq 2.1: potential emissions)
9. **AbateCost** from f, c_redist, L (Eq 1.5: abatement expenditure)
10. **μ** from AbateCost, θ₁, θ₂, E_pot (Eq 1.6: fraction of emissions abated, capped at μ_max)
11. **Λ** from AbateCost, Y_damaged (Eq 1.7: abatement cost fraction)
12. **Y_net** from Y_damaged, Λ (Eq 1.8: production after abatement costs)
13. **y_eff** from y, AbateCost, L (Eq 1.9: effective per-capita income)
14. **G_eff** from f, f_gdp, G_climate (Eq 4.4: effective Gini after redistribution/abatement; when f_gdp >= 1, G_eff = G_climate with no redistribution effect)
15. **U** from y_eff, G_eff, η (Eq 3.5: mean utility)
16. **E** from σ, μ, Y_gross (Eq 2.3: actual emissions after abatement)
17. **dK/dt** from s, Y_net, δ, K (Eq 1.10: capital tendency)
18. **dGini/dt, Gini_step** from Gini dynamics (Gini tendency and step change)

### Core Components

#### 1. Economic Model (Solow-Swann Growth)

**Eq. (1.1) - Production Function (Cobb-Douglas):**
```
Y_gross(t) = A(t) · K(t)^α · L(t)^(1-α)
```

**Eq. (1.2) - Income-Dependent Climate Damage:**

**Income Distribution:**
For a Pareto income distribution with parameter `a > 1`:
```
y(F) = ȳ · (1 - 1/a) · (1-F)^(-1/a),  F ∈ [0,1]
```
where `F` is the population fraction (poorest), `ȳ` is mean income, and pre-damage Gini is `G₀ = 1/(2a-1)`.

**Damage Function (Half-Saturation Model):**
```
ω_max(ΔT) = psi1 · ΔT + psi2 · ΔT²  [Barrage & Nordhaus 2023]
ω(y) = ω_max · y_damage_halfsat / (y_damage_halfsat + y)
```
where:
- `ω_max` is the maximum damage fraction (applies at zero income)
- `y_damage_halfsat` is the income level at which damage equals ω_max/2
- At income `y = 0`: damage = `ω_max` (maximum for poorest)
- At income `y = y_damage_halfsat`: damage = `ω_max/2`
- As income `y → ∞`: damage → 0 (wealthy largely protected)

**Analytical Solution for Aggregate Damage:**
The aggregate damage (fraction of total production lost) is computed analytically:
```
b = y_damage_halfsat · a / (ȳ · (a-1))
Ω = (1/ȳ) · ∫₀¹ ω(y(F)) · y(F) · dF
  = ω_max · (y_damage_halfsat/ȳ) · ₂F₁(1, a, a+1, -b)
```
where:
- `b` is a dimensionless damage concentration parameter
- `₂F₁` is the Gauss hypergeometric function

**Post-Damage Inequality (Gini Effect):**
Climate damage increases inequality because lower-income populations suffer proportionally greater losses. The post-damage Gini coefficient `G_climate` is computed using:
```
b = y_damage_halfsat · a / (ȳ · (a-1))               [dimensionless damage parameter]
G₀ = 1/(2a-1)                                        [pre-damage Gini]
Φ = ₂F₁(a-1, 1, a, -b)                               [mean damage factor]
H = ₂F₁(1, 2a-1, 2a, -b)                             [Gini adjustment factor]
ω_mean = ω_max · Φ                                   [mean damage across distribution]

G_climate = 1 - (1 - G₀) · (1 - ω_max · H) / (1 - ω_mean)
```

**Physical Interpretation:**
- As `y_damage_halfsat → ∞`: damage becomes uniform, `G_climate → G₀` (no inequality effect)
- As `y_damage_halfsat → 0`: damage is maximally regressive (concentrated on poor)
- As `ΔT → 0`: `ω_max → 0` and `Ω → 0` (no damage)

**Implementation:**
All integrals are solved analytically using closed-form solutions based on the Gauss hypergeometric function (₂F₁) from `scipy.special.hyp2f1`. This avoids numerical integration and is exact within numerical precision (~1e-16). The scipy implementation provides 200x+ speedup compared to arbitrary-precision libraries while maintaining excellent accuracy for this application. See `climate_damage_distribution.py` for complete derivations and implementation.

**Eq. (1.3) - Damaged Production:**
```
Y_damaged(t) = (1 - Ω(t)) · Y_gross(t)
```
This is production after accounting for climate damage but before abatement costs.

**Eq. (1.4) - Mean Per-Capita Income:**
```
y(t) = (1 - s) · Y_damaged(t) / L(t)
```

**Eq. (1.5) - Abatement Cost:**
```
AbateCost(t) = f · c_redist(t) · L(t)
```
This is the total amount society allocates to emissions abatement, where:
- `f` = fraction of redistributable resources allocated to abatement (0 ≤ f ≤ 1)
- `c_redist(t)` = per-capita amount of income available for redistribution
- `L(t)` = population

**Eq. (1.6) - Abatement Fraction:**
```
μ(t) = min(μ_max, [AbateCost(t) · θ₂ / (E_pot(t) · θ₁(t))]^(1/θ₂))
```
The fraction of potential emissions that are abated, where:
- `E_pot(t) = σ(t) · Y_gross(t)` = potential (unabated) emissions
- `θ₁(t)` = marginal cost of abatement as μ→1 ($ tCO₂⁻¹)
- `θ₂` = abatement cost exponent (θ₂=2 gives quadratic cost function)
- `μ_max` = maximum allowed abatement fraction (cap on μ)

The calculated μ is capped at μ_max. Values of μ_max > 1 allow for carbon dioxide removal (negative emissions). If μ_max is not specified in the configuration, it defaults to INVERSE_EPSILON (effectively no cap).

This formulation differs from Nordhaus in that reducing carbon intensity σ(t) reduces the cost of abating remaining emissions, since there are fewer emissions to abate.

**Eq. (1.7) - Abatement Cost Fraction:**
```
Λ(t) = AbateCost(t) / Y_damaged(t)
```
This represents the fraction of damaged production allocated to emissions abatement.

**Eq. (1.8) - Net Production:**
```
Y_net(t) = (1 - Λ(t)) · Y_damaged(t)
```
Production after both climate damage and abatement costs.

**Eq. (1.9) - Effective Per-Capita Income:**
```
y_eff(t) = y(t) - AbateCost(t) / L(t)
```
This is the per-capita income after subtracting abatement costs, used for utility calculations.

**Eq. (1.10) - Capital Accumulation:**
```
dK/dt = s · Y_net(t) - δ · K(t)
```

#### 2. Climate Model

**Eq. (2.1) - Potential Emissions:**
```
E_pot(t) = σ(t) · Y_gross(t)
```
This is the emissions rate without any abatement.

**Eq. (2.2) - Temperature Change:**
```
ΔT(t) = k_climate · ∫₀^t E(t') dt'
       = k_climate · Ecum(t)
```
Temperature change is proportional to cumulative carbon dioxide emissions.

**Eq. (2.3) - Actual Emissions:**
```
E(t) = σ(t) · (1 - μ(t)) · Y_gross(t)
     = (1 - μ(t)) · E_pot(t)
```
This is the actual emissions rate after abatement.

#### 3. Income Distribution and Utility

**Eq. (3.1) - Pareto-Lorenz Distribution:**
```
ℒ(F) = 1 - (1 - F)^(1-1/a)
```

where `F` is the fraction of the population with the lowest incomes.

**Eq. (3.2) - Gini Index:**
```
G = 1/(2a - 1)
a = (1 + 1/G)/2
```

**Eq. (3.3) - Income at Rank F:**
```
c(F) = y · (1 - 1/a) · (1 - F)^(-1/a)
```

**Eq. (3.4) - Isoelastic Utility Function (CRRA):**
```
u(c) = (c^(1-η) - 1)/(1 - η)  for η ≠ 1
u(c) = ln(c)                   for η = 1
```

where `η` is the coefficient of relative risk aversion.

**Eq. (3.5) - Mean Population Utility:**
```
U = [y^(1-η)/(1-η)] · [(1+G)^η(1-G)^(1-η)/(1+G(2η-1))]^(1/(1-η))  for η ≠ 1
U = ln(y) + ln((1-G)/(1+G)) + 2G/(1+G)                              for η = 1
```

#### 4. Redistribution Mechanics

The model considers allocation of resources between income redistribution and emissions abatement. The key parameters are:
- `G₁` = initial Gini index
- `f_gdp` = fraction of total income to be redistributed (specified exogenously)
- `f` = fraction of redistributable resources allocated to abatement (0 ≤ f ≤ 1)

**Eq. (4.1) - Fraction of Income Redistributed:**

Given `f_gdp` and `G₁`, we numerically solve for `G₂` (the Gini index after full redistribution) using the relationship:
```
f_gdp(F*) = [2(G₁-G₂)/(1-G₁)(1+G₂)] · [((1+G₁)(1-G₂))/((1-G₁)(1+G₂))]^((1+G₁)(1-G₂)/(2(G₂-G₁)))
```
where `F*` is the crossing rank (see Eq. 4.2).

**Eq. (4.2) - Crossing Rank:**

The population rank where income remains unchanged during redistribution:
```
F* = 1 - [((1+G₁)(1-G₂))/((1-G₁)(1+G₂))]^(((1+G₁)(1+G₂))/(2(G₂-G₁)))
```

**Eq. (4.3) - Per-Capita Amount Redistributed:**
```
c_redist = y · f_gdp
```
where `y` is mean per-capita income.

**Eq. (4.4) - Effective Gini Index:**

When fraction `f` of redistributable resources goes to abatement instead of redistribution, the effective Gini index is calculated using a two-step Pareto-preserving approach (see `income_distribution.calculate_Gini_effective_redistribute_abate`).

For reference, the formula is:
```
G_eff(f) = (1-f_gdp)/(1-f·f_gdp) · [1 - (1 - G₁)^((1-f_gdp(1-F*))/(1-f_gdp))]
```

where:
- `f = 0`: all resources go to redistribution → `G_eff(0)` = minimum (most equal)
- `f = 1`: all resources go to abatement → `G_eff(1)` = maximum Gini given removal
- `0 < f < 1`: mixed allocation

**Fraction of Emissions Abated:**

See Eq. (1.6) above. The abatement fraction is determined by the amount society allocates to abatement relative to potential emissions and the marginal abatement cost.

**Enhanced Redistribution Mode (f_gdp >= 1) - Partial Implementation**

The model now supports `f_gdp >= 1` with special handling that disables redistribution and allows pure abatement optimization.

**Current Behavior (f_gdp < 1):**
- Redistribution operates within the Pareto family of distributions
- Income transfers preserve the general shape of the distribution
- The Gini coefficient changes, but the underlying distribution remains Pareto
- Climate damage calculations assume a fixed Pareto distribution with parameter `a` derived from current Gini
- Control variable `f` determines allocation between abatement and redistribution

**Implemented Behavior (f_gdp >= 1):**
When `f_gdp >= 1`, the model disables redistribution and enables pure abatement optimization:

1. **Redistribution Disabled** (`economic_model.py:164-167`):
   - Effective Gini is set equal to climate-damaged Gini: `G_eff = Gini_climate`
   - No redistribution effect on inequality (bypasses `calculate_Gini_effective_redistribute_abate`)
   - Gini evolves only through climate damage and restoration dynamics

2. **Climate Damage Calculation** (`climate_damage_distribution.py:156`):
   - Triggers uniform damage approximation: `Omega = omega_max`
   - Preserves Gini unchanged: `Gini_climate = Gini_current`
   - Rationale: Structural redistribution (`delta_L >= 1`) invalidates Pareto distribution assumption
   - Falls back to simple uniform damage rather than attempting income-dependent calculation

3. **Abatement Budget Mechanics**:
   - Available budget: `redistribution = y * delta_L` (Line 136 of `economic_model.py`)
   - With `delta_L >= 1`, this creates `redistribution >= y` (budget at least equals full per-capita income)
   - Abatement expenditure: `AbateCost = f * redistribution * L` (Line 142)
   - Effective income: `y_eff = y - AbateCost/L = y - f * redistribution`

4. **Optimizer Behavior**:
   - The optimizer chooses `f` to maximize utility over time
   - **Naturally selects `f << 1`** because:
     - Large `f` would make `y_eff = y - f * redistribution` very small or negative
     - This would result in terrible current utility (Consumption crash)
     - Optimizer balances current Consumption vs. future climate benefits
   - **Equivalence**: Optimization of `f` becomes equivalent to optimizing the abatement/Consumption tradeoff
   - No redistribution component in utility calculation (since `G_eff = Gini_climate`)

5. **Physical Interpretation**:
   - `f_gdp >= 1` represents a model mode where redistribution is turned off
   - Allows studying pure abatement policy without redistribution considerations
   - Budget parameter `f_gdp` scales the available resources, but optimizer self-limits via utility constraints
   - Climate damage treated as uniform across income levels (first-order approximation)

**Implementation Status**:
- ✓ Redistribution disabled in `economic_model.py` (lines 164-167)
- ✓ Uniform damage approximation in `climate_damage_distribution.py` (line 156)
- ✓ Uses `INVERSE_EPSILON` constant from `constants.py` (no hardcoded values)
- ✓ All existing unit tests pass

**Future Enhancements**:
For more sophisticated treatment of `f_gdp >= 1` with non-Pareto income distributions, see **Next Steps, Section 2**.

#### 5. Gini Index Dynamics and Persistence

The Gini index is now a **state variable** that evolves over time, allowing for persistence of redistribution effects and gradual restoration to baseline inequality.

**State Variable:**
```
Gini(t) - Current Gini index of the income distribution
```

**Gini Evolution:**

The Gini index evolves through two mechanisms:

1. **Instantaneous Step Change** (fraction of policy effect applied immediately):
```
Gini_step = Gini_fract · (G_eff - Gini)
```
where:
- `G_eff` is the effective Gini from current policy (redistribution/abatement allocation)
- `Gini_fract` is the fraction of the change applied as an immediate step (0 ≤ Gini_fract ≤ 1)
- `Gini_fract = 0`: no immediate effect (fully persistent system)
- `Gini_fract = 1`: full immediate effect (no persistence)
- `Gini_fract = 0.1`: 10% of policy effect occurs immediately

2. **Continuous Restoration** (gradual return to baseline):
```
dGini/dt = -Gini_restore · (Gini - Gini_initial)
```
where:
- `Gini_restore` is the restoration rate (yr⁻¹)
- `Gini_restore = 0`: no restoration (persistent policy effects)
- `Gini_restore > 0`: gradual restoration toward initial inequality
- `Gini_restore = 0.1`: 10% per year restoration rate (timescale ~10 years)

**Combined Update Rule:**
```
Gini(t+dt) = Gini(t) + dt · dGini/dt + Gini_step
```

**Physical Interpretation:**

This formulation captures two competing effects:
- **Policy pressure** (via `Gini_step`): Redistribution policies push toward lower inequality (G_eff < Gini_initial)
- **Structural restoration** (via `dGini/dt`): Absent continued intervention, inequality tends to return to baseline levels

The `Gini_fract` parameter controls the **speed of policy effect**:
- Small `Gini_fract`: Policy effects build up gradually (high persistence/inertia)
- Large `Gini_fract`: Policy effects manifest quickly (low persistence/inertia)

The `Gini_restore` parameter controls the **persistence of achieved changes**:
- Small `Gini_restore`: Changes are long-lasting
- Large `Gini_restore`: Changes decay quickly without continued policy pressure

**Climate Damage Interaction:**

Climate damage affects inequality through the intermediate variable `G_climate`:
```
Current Gini → (climate damage) → G_climate → (redistribution/abatement) → G_eff
```
where `G_climate > Gini` due to regressive climate damage impacts (lower incomes suffer proportionally more).

## Key Parameters

Parameters are organized into groups as specified in the JSON configuration file.

### Scalar Parameters (Time-Invariant)

Economic parameters:

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| `α` | Output elasticity of capital (capital share of income) | - | `alpha` |
| `δ` | Capital depreciation rate | yr⁻¹ | `delta` |
| `s` | Savings rate (fraction of net production saved) | - | `s` |

Climate and abatement parameters:

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| `psi1` | Linear climate damage coefficient [Barrage & Nordhaus 2023] | °C⁻¹ | `psi1` |
| `psi2` | Quadratic climate damage coefficient [Barrage & Nordhaus 2023] | °C⁻² | `psi2` |
| `y_damage_halfsat` | Income half-saturation for climate damage (lower = more regressive) | $ | `y_damage_halfsat` |
| `k_climate` | Temperature sensitivity to cumulative emissions | °C tCO₂⁻¹ | `k_climate` |
| `θ₂` | Abatement cost exponent (controls cost curve shape) | - | `theta2` |
| `μ_max` | Maximum allowed abatement fraction (cap on μ). Values >1 allow carbon removal. Defaults to INVERSE_EPSILON (no cap) if omitted. | - | `mu_max` |
| `Ecum_initial` | Initial cumulative CO2 emissions. Defaults to 0.0 (no prior emissions) if omitted. | tCO₂ | `Ecum_initial` |

Utility and inequality parameters:

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| `η` | Coefficient of relative risk aversion (CRRA) | - | `eta` |
| `ρ` | Pure rate of time preference | yr⁻¹ | `rho` |
| `G₁` | Initial Gini index (0 = perfect equality, 1 = max inequality) | - | `Gini_initial` |
| `Gini_fract` | Fraction of effective Gini change as instantaneous step (0 = no step, 1 = full step) | - | `Gini_fract` |
| `Gini_restore` | Rate at which Gini restores to initial value (0 = no restoration) | yr⁻¹ | `Gini_restore` |
| `f_gdp` | Fraction of income available for redistribution (<1: active redistribution; >=1: redistribution disabled, pure abatement mode) | - | `fract_gdp` |

### Time-Dependent Functions

These functions are evaluated at each time step:

| Function | Description | Units | JSON Key |
|----------|-------------|-------|----------|
| `A(t)` | Total factor productivity | - | `A` |
| `L(t)` | Population | people | `L` |
| `σ(t)` | Carbon intensity of GDP | tCO₂ $⁻¹ | `sigma` |
| `θ₁(t)` | Marginal abatement cost as μ→1 | $ tCO₂⁻¹ | `theta1` |

Each function is specified by `type` and type-specific parameters (e.g., `initial_value`, `growth_rate`). Six function types are available: `constant`, `exponential_growth`, `logistic_growth`, `piecewise_linear`, `double_exponential_growth` (Barrage & Nordhaus 2023), and `gompertz_growth` (Barrage & Nordhaus 2023). See the Configuration section below for detailed specifications.

### Control Variables

The model supports two control variables that can be optimized:

| Variable | Description | Units | JSON Key | Status |
|----------|-------------|-------|----------|---------|
| `f(t)` | Fraction of redistributable resources allocated to abatement | - | `control_function` | Required |
| `s(t)` | Savings rate (fraction of net production invested) | - | `s` in `time_functions` or `s_control_function` | Required |

**f(t) - Abatement Allocation:** Determines the allocation between emissions abatement and income redistribution (0 = all to redistribution, 1 = all to abatement). Always specified via `control_function` in the configuration.

**s(t) - Savings Rate:** Determines the fraction of net output allocated to investment vs. Consumption. Can be specified in two ways:
- **Fixed/Prescribed s(t):** Defined in `time_functions['s']` using any time function type (constant, piecewise_linear, etc.)
- **Optimized s(t):** Defined in `s_control_function` to enable dual optimization of both f(t) and s(t)

When both `control_function` and `s_control_function` are present, the model operates in **dual optimization mode**, allowing simultaneous optimization of the abatement-redistribution tradeoff and the Consumption-investment tradeoff.

### Integration Parameters

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| `t_start` | Start time for integration | yr | `t_start` |
| `t_end` | End time for integration | yr | `t_end` |
| `dt` | Time step for Euler integration | yr | `dt` |
| `rtol` | Relative tolerance (reserved for future use) | - | `rtol` |
| `atol` | Absolute tolerance (reserved for future use) | - | `atol` |

### Initial Conditions (Computed Automatically)

| Variable | Value | Description |
|----------|-------|-------------|
| `K(0)` | `(s·A(0)/δ)^(1/(1-α))·L(0)` | Steady-state capital stock |
| `Ecum(0)` | `0` | No cumulative emissions at start |

## Model Features

### Simplifying Assumptions

To maintain analytical tractability:
- Fixed Pareto-Lorenz income distribution (parameterized by Gini index)
- Proportional relationship between temperature and cumulative emissions
- Power-law relationships for climate damage and abatement costs
- No distinction between population and labor force
- Exogenous technological progress `A(t)` and population `L(t)`

### Key Insights

1. **Redistribution vs. Climate Action Tradeoff**: Resources allocated to income redistribution provide immediate utility gains (especially with high `η`), while emissions abatement provides future benefits by reducing climate damage.

2. **Diminishing Marginal Utility**: Higher values of `η` mean that redistributing income from rich to poor has greater utility benefits, favoring redistribution over abatement.

3. **Time Preference**: Higher discount rates (`ρ`) favor immediate redistribution over long-term climate benefits.

4. **Income Distribution Mechanics**: Taxing the wealthy reduces the Gini index even if revenues are allocated to abatement rather than redistribution, but only redistribution increases current aggregate utility.

## Implementation: Key Functions

The `income_distribution.py` module provides the core mathematical functions for calculating income distribution metrics and effective Gini indices under different allocation scenarios.

### Basic Conversion Functions

- **`a_from_G(G)`** - Converts Gini index to Pareto distribution parameter `a` using equation (4)
- **`L_pareto(F, G)`** - Calculates Lorenz curve value at population fraction `F` for a given Gini index (equation 2)

### Redistribution Mechanics

- **`crossing_rank_from_G(Gini_initial, G2)`** - Computes the population rank `F*` where income remains unchanged during redistribution from `Gini_initial` to `G2` (equation 10)

### Inverse Problem: Finding G2 from f_gdp

- **`_phi(r)`** - Helper function for numerical root finding; computes `φ(r) = (r-1) · r^(1/(r-1)-1)` with proper handling of edge cases

- **`G2_from_deltaL(deltaL, Gini_initial)`** - **Solves the inverse problem**: given an initial Gini `Gini_initial` and a desired redistribution amount `f_gdp`, numerically finds the target Gini `G2` that would result from full redistribution. Uses `scipy.optimize.root_scalar` with Brent's method. Returns `(G2, remainder)` where remainder is non-zero if `f_gdp` exceeds the maximum possible for the Pareto family (caps at G2=0).

### Effective Gini Calculation

- **`calculate_Gini_effective_redistribute_abate(f, deltaL, Gini_initial)`** - **Main function** that calculates the effective Gini index when fraction `f` of redistributable resources is allocated to emissions abatement instead of redistribution.

  **Algorithm:**
  1. Solves for full-redistribution target `G2` from `f_gdp` and `Gini_initial`
  2. Computes crossing rank `F*` for the `(Gini_initial → G2)` transition
  3. Calculates effective redistribution amount `f_gdp_eff` at the same `F*` for partial allocation
  4. Solves for Pareto-equivalent `G2_eff` from `f_gdp_eff`

  **Parameters:**
  - `f = 0`: All resources to redistribution → minimum Gini (maximum equality)
  - `f = 1`: All resources to abatement → maximum Gini given removal
  - `0 < f < 1`: Mixed allocation

  **Returns:** `(G2_eff, remainder)` tuple

### Usage Example

```python
from income_distribution import calculate_Gini_effective_redistribute_abate

# Initial Gini index
Gini_initial = 0.4

# Fraction of income to be redistributed (e.g., 5% of total income)
deltaL = 0.05

# Fraction allocated to abatement vs redistribution
f = 0.5  # 50% to abatement, 50% to redistribution

# Calculate effective Gini index
G_eff, remainder = calculate_Gini_effective_redistribute_abate(f, deltaL, Gini_initial)

print(f"Effective Gini: {G_eff:.4f}")
```

## Parameter Organization

The model uses JSON configuration files to specify all parameters. Configuration is loaded via `load_configuration(config_path)` in `parameters.py`.

### Configuration File Structure

Each JSON configuration file must contain:

1. **`run_name`** - String identifier used for output directory naming
2. **`description`** - Optional description of the scenario
3. **`scalar_parameters`** - Time-invariant model constants:
   - Economic: `alpha`, `delta`, `s`
   - Climate: `psi1`, `psi2`, `y_damage_halfsat`, `k_climate`
   - Utility: `eta`, `rho`
   - Distribution: `Gini_initial`, `Gini_fract`, `Gini_restore`, `fract_gdp`

4. **`time_functions`** - Time-dependent functions (A, L, sigma, theta1), each specified with:
   - `type`: One of six available function types (see details below)
   - Type-specific parameters (e.g., `initial_value`, `growth_rate`)

   **Available Time Function Types:**

   a. **`constant`** - Returns fixed value for all times
      - Parameters: `value`
      - Equation: `f(t) = value`

   b. **`exponential_growth`** - Exponential growth or decay
      - Parameters: `initial_value`, `growth_rate`
      - Equation: `f(t) = initial_value · exp(growth_rate · t)`

   c. **`logistic_growth`** - S-curve growth approaching asymptotic limit
      - Parameters: `L0` (initial), `L_inf` (limit), `growth_rate`
      - Equation: `f(t) = L_inf / (1 + ((L_inf/L0) - 1) · exp(-growth_rate · t))`

   d. **`piecewise_linear`** - Linear interpolation between discrete points
      - Parameters: `time_points` (array), `values` (array)
      - Equation: Linear interpolation between (time_points, values)

   e. **`double_exponential_growth`** - Weighted sum of two exponentials (Barrage & Nordhaus 2023)
      - Parameters: `initial_value`, `growth_rate_1`, `growth_rate_2`, `fract_1`
      - Equation: `f(t) = initial_value · (fract_1 · exp(growth_rate_1 · t) + (1 - fract_1) · exp(growth_rate_2 · t))`
      - **Purpose**: Models carbon intensity (sigma) with fast initial decline transitioning to slower long-term decline
      - **Typical values**: Curve fit to DICE2023 parameters:
        - `growth_rate_1 = -0.015` (fast initial decarbonization)
        - `growth_rate_2 = -0.005` (slower asymptotic decline)
        - `fract_1 = 0.70` (70% weight on fast decline)

   f. **`gompertz_growth`** - Gompertz growth function (continuous form of Barrage & Nordhaus 2023 finite-difference model)
      - Parameters: `initial_value`, `final_value`, `adjustment_coefficient`
      - Equation: `L(t) = final_value · exp(ln(initial_value / final_value) · exp(adjustment_coefficient · t))`
      - **Purpose**: Models population growth approaching asymptotic limit
      - **Properties**: At t=0: L(0) = initial_value; as t→∞: L(t) → final_value (for negative adjustment_coefficient)
      - **Note**: This form using exp/log has better numerical properties than the equivalent power form `(initial_value / final_value)^exp(...)`
      - **Typical values**: Based on DICE2023 parameters:
        - `initial_value = 7.0e9` (7 billion people)
        - `final_value = 10.0e9` (10 billion asymptotic limit)
        - `adjustment_coefficient = -0.02` (controls approach rate to limit)

5. **`integration_parameters`** - Solver configuration:
   - `t_start`, `t_end`, `dt`, `rtol`, `atol`

6. **`control_function`** - Allocation policy f(t):
   - `type`: "constant" or "piecewise_constant"
   - Type-specific parameters (e.g., `value` for constant)

See `config_baseline.json` for extensive examples of documentation.

### Initial Conditions

Initial conditions are **computed automatically**:

- **`Ecum(0) = Ecum_initial`**: Initial cumulative emissions from configuration (defaults to 0.0 if not specified)
- **`Gini(0) = Gini_initial`**: Initial Gini index from configuration
- **`K(0)`**: Initial capital stock accounting for climate damage and abatement costs

**Initial Capital Stock Calculation:**

The initial capital stock K(0) is determined iteratively to ensure consistency with climate damage from initial cumulative emissions and the initial abatement allocation. This differs from a simple steady-state calculation because:

1. **Climate damage reduces net production**: With initial temperature change ΔT(0) = k_climate · Ecum_initial, the economy experiences damage Ω that reduces effective output available for capital accumulation.

2. **Abatement costs reduce net production**: The initial control policy f(0) allocates resources to abatement, further reducing net production available for capital accumulation.

**Iterative Solution Method:**

Starting from the steady-state relationship:
```
K₀ = [(s · (1 - Ω) · (1 - λ) · A(0)) / δ]^(1/(1-α)) · L(0)
```
where:
- `Ω` = aggregate climate damage fraction (depends on K₀ through y_gross)
- `λ = (1-s) · f(0) · f_gdp` = abatement cost fraction

The algorithm iterates until Ω converges (tolerance: 1e-12):

1. **Initialize**: Compute initial estimate of Ω from ΔT and typical income
2. **Iterate**:
   - Calculate K₀ using current Ω estimate
   - Calculate y_gross = A(0) · (K₀^α) · (L(0)^(1-α)) / L(0)
   - Update Ω from income-dependent climate damage function
   - Check convergence: |Ω_new - Ω_old| < ε
3. **Converge**: Typically converges in 2-5 iterations

**Special Cases:**
- If `Ecum_initial = 0`: Converges immediately (Ω = 0, no climate damage)
- If `f(0) = 0`: No abatement cost (λ = 0)
- If both zero: Reduces to simple steady state K₀ = (s·A(0)/δ)^(1/(1-α)) · L(0)

This initialization ensures the model starts from an economically consistent state that accounts for:
- Pre-existing climate change impacts on production and inequality
- Initial policy allocation between abatement and redistribution
- Income-dependent climate vulnerability of the population

The iteration count is printed during model execution for debugging purposes.

### Example Configuration

See `config_baseline.json` for a complete example. To create new scenarios, copy and modify this file.

**Note**: `config_test_DICE.json` provides a configuration for simulations close to the parameters and setup presented in Barrage & Nordhaus (2023), including Gompertz population growth, double exponential functions for carbon intensity and abatement costs, and settings that replicate DICE2023 behavior (deltaL = 1.0 for pure abatement mode, Gini_initial = 0.0 for no inequality).

**Example: Population with Gompertz growth**
```json
"L": {
  "type": "gompertz_growth",
  "initial_value": 7.0e9,
  "final_value": 10.0e9,
  "adjustment_coefficient": -0.02
}
```

**Example: Carbon intensity with double exponential decline**
```json
"sigma": {
  "type": "double_exponential_growth",
  "initial_value": 0.0005,
  "growth_rate_1": -0.015,
  "growth_rate_2": -0.005,
  "fract_1": 0.70
}
```

**Example: TFP with simple exponential growth**
```json
"A": {
  "type": "exponential_growth",
  "initial_value": 454.174,
  "growth_rate": 0.01
}
```

### Loading Configuration

```python
from parameters import load_configuration

config = load_configuration('config_baseline.json')
# config.run_name contains the run identifier
# config.scalar_params, config.time_functions, etc. are populated
```

The `evaluate_params_at_time(t, config)` helper combines all parameters into a dict for use with `calculate_tendencies()`.

## Unit Testing: Validating Analytical Solutions

The project includes unit tests that validate the analytical solutions for key model equations by comparing them against high-precision numerical integration.

### Unit Test for Equation (1.2): Climate Damage

The file `unit_test_eq1.2.py` validates the analytical solution for aggregate climate damage (Ω) and post-damage Gini coefficient (G_climate).

**What it tests:**

The analytical solution uses hypergeometric functions to compute:
```
Ω = ω_max · (y_damage_halfsat/ȳ) · ₂F₁(1, a, a+1, -b)
```
where `b = y_damage_halfsat · a / (ȳ · (a-1))`

This is compared against high-precision numerical integration of the original integral:
```
Ω = (1/ȳ) · ∫₀¹ ω(y(F)) · y(F) · dF
```

**Running the test:**

```bash
python unit_test_eq1.2.py
```

**Expected output:**

The test generates 10 random parameter combinations covering a wide range of:
- Gini indices (inequality levels): 0.2 to 0.7
- Mean incomes: $1,000 to $100,000
- Half-saturation incomes: $1 to $10,000
- Maximum damage fractions: 5% to 30%

For each case, it prints:
- Parameter values (G, ȳ, k, ω_max)
- Analytical solution result
- Numerical integration result
- Relative error
- PASS/FAIL status (tolerance: 1e-9)

**Example output:**
```
================================================================================
Unit Test: Equation (1.2) - Climate Damage Analytical Solution
================================================================================

Validating analytical hypergeometric solution against numerical integration
Target tolerance: 1e-9 relative error

Case  1:  G=0.5488  ȳ= 28183.8  k=  8377.4  ω_max=0.2382
          Ω_analytical = 0.106573645123
          Ω_numerical  = 0.106573645123
          Rel. error   = 1.23e-12  ✓ PASS

[... 9 more cases ...]

================================================================================
All 10 test cases PASSED
Maximum relative error: 4.56e-11
================================================================================
```

**Interpretation:**

- **PASS**: The analytical solution matches numerical integration to within 1e-9 relative tolerance, confirming the hypergeometric formula is correctly derived and implemented.
- **Maximum relative error**: Typically ~1e-10 to 1e-12, demonstrating excellent agreement between analytical and numerical approaches.

**Technical details:**

- Production code uses `scipy.special.hyp2f1` for optimal performance (200x+ faster than arbitrary-precision alternatives)
- Unit tests use `mpmath` library for arbitrary-precision arithmetic (80 decimal places) to validate accuracy
- Numerical integration in tests performed with `mpmath.quad()` adaptive quadrature
- Tests validate both the aggregate damage (Ω) and implicitly the underlying income distribution formulas
- Random seed fixed for reproducibility

**Purpose:**

This unit test provides confidence that:
1. The analytical derivation of the hypergeometric solution is mathematically correct
2. The implementation in `climate_damage_distribution.py` correctly evaluates the formulas
3. The solution is numerically stable across a wide range of realistic parameter values

### Testing the Forward Model

The project includes a comprehensive test script to verify the forward model integration and demonstrate the complete workflow from configuration loading through output generation.

#### Quick Start

To test the model with the baseline configuration:

```bash
python test_integration.py config_baseline.json
```

This command will:
1. Load the baseline configuration from `config_baseline.json`
2. Display key model parameters and setup information
3. Run the forward integration over the specified time period
4. Show detailed results summary (initial state, final state, changes)
5. Generate timestamped output directory with CSV data and PDF plots

#### Command Line Usage

The test script requires a configuration file argument:

```bash
python test_integration.py <config_file>
```

**Examples:**
```bash
# Test with baseline scenario
python test_integration.py config_baseline.json

# Test with high inequality scenario
python test_integration.py config_high_inequality.json

# Test with custom configuration
python test_integration.py my_custom_config.json
```

If you run the script without arguments, it will display usage instructions.

#### Understanding the Output

The test script provides detailed console output including:

- **Configuration Summary**: Run name, time span, key parameters
- **Integration Progress**: Confirmation of successful model execution
- **Results Summary**:
  - Initial state (t=0): all key variables at start
  - Final state (t=end): all key variables at end of simulation
  - Changes: percentage and absolute changes over simulation period
- **Output Files**: Paths to generated CSV and PDF files

#### Generated Files

Each test run creates a timestamped directory:
```
./data/output/{run_name}_YYYYMMDD-HHMMSS/
├── results.csv    # Complete time series data (all variables)
└── plots.pdf      # Multi-page charts organized by variable type
```

The PDF contains four organized sections:
1. **Dimensionless Ratios** - Policy variables and summary outcomes
2. **Dollar Variables** - Economic flows and stocks
3. **Physical Variables** - Climate and emissions data
4. **Specified Functions** - Exogenous model inputs

#### Testing Different Scenarios

Create new test scenarios by copying and modifying configuration files:

```bash
# Copy baseline configuration
cp config_baseline.json config_my_test.json

# Edit parameters in config_my_test.json
# Then test with:
python test_integration.py config_my_test.json
```

This testing framework validates the complete model pipeline and provides immediate visual feedback on model behavior through the generated charts.

### Testing the Optimization with Parameter Overrides

The optimization test script supports command line parameter overrides, enabling automated parameter sweeps without creating multiple configuration files.

#### Command Line Override Syntax

Override any configuration parameter using dot notation:

```bash
python test_optimization.py config.json --key.subkey.value new_value
```

**Examples:**

```bash
# Override single parameter
python test_optimization.py config_baseline.json --scalar_parameters.alpha 0.35

# Override multiple parameters
python test_optimization.py config_baseline.json \
  --run_name "sensitivity_test" \
  --optimization_parameters.initial_guess 0.3 \
  --scalar_parameters.rho 0.015

# Override nested parameters
python test_optimization.py config_baseline.json \
  --time_functions.A.growth_rate 0.02 \
  --optimization_parameters.n_points_final 100
```

**Common overrides:**
- `--run_name <name>` - Set output directory name
- `--scalar_parameters.alpha <value>` - Capital share
- `--scalar_parameters.rho <value>` - Time preference rate
- `--scalar_parameters.eta <value>` - Risk aversion coefficient
- `--optimization_parameters.initial_guess <value>` - Starting point
- `--optimization_parameters.max_evaluations <value>` - Iteration budget
- `--optimization_parameters.n_points_final <value>` - Target control points
- `--time_functions.A.growth_rate <value>` - TFP growth rate

#### Automated Parameter Sweeps

The `run_initial_guess_sweep.py` script demonstrates automated testing across multiple parameter values:

```bash
python run_initial_guess_sweep.py config_baseline.json
```

This runs optimization 11 times with `initial_guess` values from 0.0 to 1.0 (step 0.1), automatically creating separate output directories for each run.

**Creating custom sweep scripts:**

```python
import subprocess

config_file = "config_baseline.json"

# Sweep over alpha values
for alpha in [0.25, 0.30, 0.35, 0.40]:
    cmd = [
        "python", "test_optimization.py", config_file,
        "--scalar_parameters.alpha", str(alpha),
        "--run_name", f"alpha_{alpha:.2f}"
    ]
    subprocess.run(cmd, check=True)
```

**Benefits of command line overrides:**
- No need to create dozens of nearly-identical JSON files
- Easy to script parameter sweeps in bash or Python
- Git-friendly: only baseline configs need version control
- Clear provenance: command documents what changed from baseline
- Composable: combine multiple overrides in one command

## Time Integration

The model uses Euler's method with fixed time steps for transparent integration that ensures all functional relationships are satisfied exactly at output points.

### Integration Function

```python
from economic_model import integrate_model
from parameters import load_configuration

config = load_configuration('config_baseline.json')
results = integrate_model(config)
```

The `integrate_model(config)` function:
- Uses simple Euler integration: `state(t+dt) = state(t) + dt * tendency(t)`
- Time step `dt` is specified in the JSON configuration
- Returns a dictionary containing time series for all model variables

### Implementation Notes

**Negative Emissions and Cumulative Emissions Floor:**

The model allows negative emissions E(t) (carbon removal through direct air capture, afforestation, etc.), but prevents cumulative emissions Ecum from going negative:

```python
# In integrate_model() Euler step:
state['Ecum'] = max(0.0, state['Ecum'] + dt * outputs['dEcum_dt'])
```

This ensures:
- Positive E: Normal emissions, Ecum increases
- Negative E: Carbon removal, Ecum decreases
- Floor at zero: Cannot remove more CO₂ than was ever emitted (Ecum ≥ 0)

The clamp is applied during integration rather than modifying E itself, allowing the emissions rate to reflect the model's physical calculations while preventing unphysical cumulative emissions.

### Performance Optimizations

The model includes several optimizations for computational efficiency while maintaining numerical accuracy:

**1. Income-Dependent Climate Damage Iteration (economic_model.py)**

Climate damage depends on effective per-capita income (y_eff), which itself depends on climate damage, creating a circular dependency. This is resolved iteratively:

```python
# Convergence criterion using LOOSE_EPSILON (1e-10)
converged = np.abs(y_eff - y_eff_prev) < LOOSE_EPSILON
```

- **RELAXATION_FACTOR = 1.0**: No relaxation (direct substitution) provides fastest convergence
- **Typical iterations**: 5-6 per timestep (down from ~45 with relaxation)
- **Initial guess**: Analytical approximation using current state adapts better than using previous timestep
- **Convergence tolerance**: LOOSE_EPSILON (1e-10) balances speed and precision

**2. Hypergeometric Function Evaluation (climate_damage_distribution.py)**

The analytical climate damage solution requires evaluating the Gauss hypergeometric function ₂F₁:

```python
from scipy.special import hyp2f1
H1 = hyp2f1(1.0, a, a + 1.0, -b)  # Mean damage factor
H2 = hyp2f1(1.0, 2.0 * a, 2.0 * a + 1.0, -b)  # Inequality adjustment
```

- **scipy.special.hyp2f1**: ~200x faster than arbitrary-precision libraries
- **Accuracy**: Machine precision (~1e-16 relative error)
- **Performance**: Evaluated twice per timestep per y_eff iteration

**3. Numerical Constants (constants.py)**

Two precision levels for different purposes:

- **EPSILON = 1e-12**: Strict tolerance for mathematical comparisons (Gini bounds, float comparisons)
- **LOOSE_EPSILON = 1e-10**: Practical tolerance for iterative solvers and optimization convergence
  - Used for y_eff convergence in economic_model.py
  - Default value for xtol_abs in optimization (control parameter convergence)
  - Appropriate for variables in [0, 1] range

**Cumulative Speedup:**

Compared to initial implementation, these optimizations provide ~200x faster integration:
- 400-year simulation: ~0.05 seconds (vs ~12 seconds originally)
- Full optimization (10,000 evaluations): ~10 minutes (vs ~33 hours originally)

### Output Variables

The results dictionary contains arrays for:
- **Time**: `t`
- **State variables**: `K`, `Ecum`, `Gini`
- **Time-dependent inputs**: `A`, `L`, `sigma`, `theta1`, `f`, `s`
- **Economic variables**: `Y_gross`, `Y_damaged`, `Y_net`, `y`, `y_eff`
- **Climate variables**: `delta_T`, `Omega`, `E`, `Climate_Damage`
- **Abatement variables**: `mu`, `Lambda`, `AbateCost`, `redistribution`, `marginal_abatement_cost`
- **Investment/Consumption**: `Savings`, `Consumption`
- **Inequality/utility**: `G_eff`, `Gini_climate`, `U`, `discounted_utility`
- **Tendencies**: `dK_dt`, `dEcum_dt`, `dGini_dt`, `Gini_step_change`

All arrays have the same length corresponding to time points from `t_start` to `t_end` in steps of `dt`.

## Output and Visualization

Model results are automatically saved to timestamped directories with CSV data and PDF plots.

### Saving Results

```python
from output import save_results

# After running integration
output_paths = save_results(results, config.run_name)
```

This creates a directory: `./data/output/{run_name}_YYYYMMDD-HHMMSS/`

### Output Files

**CSV File (`results.csv`):**
- Each column is a model variable
- Columns are ordered by category (time, inputs, economic flow, emissions, investment, utility, etc.)
- Each row is a time point
- First row contains variable names (header)
- Can be loaded into Excel, Python (pandas), R, etc.

**PDF File (`plots.pdf`):**
- Multi-page PDF with organized time series plots
- Each page header displays the run name for easy identification
- Variables grouped by type (dimensionless ratios, dollar variables, etc.)
- Individual plots for single variables, combined plots for related variables with legends
- Automatically uses scientific notation for large/small values

### Example Workflow

```python
from parameters import load_configuration
from economic_model import integrate_model
from output import save_results

# Load configuration
config = load_configuration('config_baseline.json')

# Run model
results = integrate_model(config)

# Save outputs
output_paths = save_results(results, config.run_name)
print(f"Results saved to: {output_paths['output_dir']}")
```

See the **Testing the Forward Model** section above for detailed instructions on using `test_integration.py`.

## Optimization Configuration

The JSON configuration supports both direct multi-point optimization and iterative refinement optimization through the `optimization_parameters` section.

### Direct Multi-Point Optimization

Specify control times and initial guesses explicitly:

```json
"optimization_parameters": {
  "max_evaluations": 10000,
  "control_times": [0, 25, 50, 75, 100],
  "initial_guess": [0.5, 0.5, 0.5, 0.5, 0.5]
}
```

**Configuration rules:**
- `control_times`: Array of times (years) where control points are placed
  - Must be in ascending order
  - For single-point optimization: `[0]`
  - For multi-point: any number of times, e.g., `[0, 25, 50, 75, 100]`
- `initial_guess`: Array of initial f values, one per control time
  - Must have same length as `control_times`
  - Each value must satisfy 0 ≤ f ≤ 1
  - For single-point: `[0.5]` (or read from `control_function.value`)
- `max_evaluations`: Maximum objective function evaluations per optimization
  - Single-point: ~1000 typically sufficient
  - Multi-point: scale with problem size (e.g., 10000 for 5 points)

### Iterative Refinement Optimization

Specify the number of refinement iterations to progressively add control points:

```json
"optimization_parameters": {
  "max_evaluations": 5000,
  "control_times": 4,
  "initial_guess": 0.5
}
```

**Configuration rules for iterative refinement:**
- `control_times`: Scalar integer specifying number of refinement iterations
  - Must be ≥ 1
- `initial_guess`: Scalar value for initial f at all control points in first iteration
  - Must satisfy 0 ≤ f ≤ 1
- `max_evaluations`: Maximum objective function evaluations per iteration
- `n_points_final`: Target number of control points in final iteration (optional)
  - If specified, the refinement base is calculated as: `base = (n_points_final - 1)^(1/(n_iterations - 1))`
  - If omitted, uses default `base = 2.0`
  - Non-integer bases prevent exact alignment with previous grids
  - Example: `n_points_final = 10` with 4 iterations gives base ≈ 2.08 → 2, 3, 5, 10 points
  - Example: default base = 2.0 with 5 iterations gives 2, 3, 5, 9, 17 points
- `xtol_abs`: Absolute tolerance on control parameters (optional, default from NLopt)
  - Recommended: `1e-10` (stops when all |Δf| < 1e-10)
  - Since f ∈ [0,1], absolute tolerance is more meaningful than relative tolerance

**Number of control points per iteration:**
- Iteration k produces `round(1 + base^(k-1))` control points
- Default base=2.0: Iteration 1: 2 points, Iteration 2: 3 points, Iteration 3: 5 points, etc.
- Custom base from n_points_final ensures the final iteration has exactly the target number of points

**Iterative refinement algorithm:**

The optimizer performs a sequence of optimizations with progressively finer control point grids. Each iteration uses the solution from the previous iteration to initialize the new optimization via PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation.

**Control point spacing - Utility-weighted distribution:**

Instead of spacing control points equally in time, points are distributed to provide approximately equal contributions to the time-discounted aggregate utility integral. This concentrates control points where they matter most for the objective function.

The control point times are calculated based on:
1. **Average TFP growth rate**: `k_A = ln(A(t_end)/A(t_start)) / (t_end - t_start)`
2. **Effective Consumption discount rate**: `r_c_effective = ρ + η · k_A · (1 - α)`
   - `ρ`: pure rate of time preference
   - `η`: coefficient of relative risk aversion
   - `α`: capital share of income
3. **Weighting discount rate**: `r_c = (r_c_effective + ρ) / 2`
   - Uses the mean of the pure rate of time preference and the effective discount rate
   - Provides a balance between pure time preference and Consumption-adjusted discounting

For iteration with N+1 control points (k = 0, 1, ..., N):
```
t(k) = -(1/r_c) · ln(1 - (k/N) · (1 - exp(-r_c · t_end)))
```

This formula ensures that each interval between control points contributes approximately equally to the discounted objective, with more points concentrated in early periods where discounting matters most. The averaging of discount rates provides a pragmatic middle ground between pure time preference and growth-adjusted discounting.

**Iteration schedule:**

- **Iteration 1**: 2 control points (k=0, 1) → `[t(0), t(1)]` = `[0, t_end]`
- **Iteration 2**: 3 control points (k=0, 1, 2)
- **Iteration 3**: 5 control points (k=0, 1, 2, 3, 4)
- **Iteration 4**: 9 control points
- **Iteration n**: 1 + 2^(n-1) control points

**Initial guess strategy:**
- **Iteration 1**: All points use `initial_guess` scalar value
- **Iteration n (n ≥ 2)**:
  - Existing points from iteration n-1 use their optimal values
  - New points use PCHIP interpolation from iteration n-1 solution
  - Interpolated values are clamped to [0, 1]

**Advantages of iterative refinement:**
- Better convergence by starting with coarse, well-initialized solutions
- Progressively captures finer temporal structure in optimal policy
- Each iteration "warm starts" from previous solution
- Avoids poor local minima that can occur with many control points from cold start
- Utility-weighted spacing focuses computational effort where it matters most
- PCHIP interpolation preserves monotonicity and shape characteristics of previous solution

### Optimization Stopping Criteria

The optimization accepts optional NLopt stopping criteria parameters:
- `xtol_abs` - Absolute tolerance on control parameters (recommended)
- `xtol_rel` - Relative tolerance on control parameters
- `ftol_abs` - Absolute tolerance on objective function
- `ftol_rel` - Relative tolerance on objective function

**Recommended practice:** Use `xtol_abs = 1e-10` as the sole stopping criterion. Since the control variable f is bounded in [0,1], absolute tolerance is more meaningful than relative tolerance, and there's no reason to want different accuracy near 0 versus near 1. The objective function can have large absolute values, making `ftol_rel` trigger prematurely even when significant improvements remain possible.

### Dual Optimization (f and s)

The model supports simultaneous optimization of both the abatement allocation fraction f(t) and the savings rate s(t). This is enabled by adding an `s_control_function` alongside the standard `control_function`.

#### Configuration for Dual Optimization

**Basic dual optimization** (constant f and s):
```json
{
  "control_function": {
    "type": "constant",
    "value": 0.5
  },
  "s_control_function": {
    "type": "constant",
    "value": 0.24
  },
  "time_functions": {
    "s": {
      "type": "constant",
      "value": 0.23974
    }
  },
  "optimization_parameters": {
    "max_evaluations": 1000,
    "control_times": [0],
    "initial_guess": [0.5],
    "s_control_times": [0],
    "s_initial_guess": [0.24],
    "algorithm": "LN_SBPLX",
    "xtol_abs": 1e-10
  }
}
```

**Notes:**
- `s_control_function` enables dual optimization mode
- `time_functions['s']` is still required as a fallback but will be overridden by `s_control_function`
- When `s_control_function` is present, optimization will jointly optimize both f and s
- Both variables use the same NLopt algorithm and stopping criteria

#### Multi-Point Dual Optimization

f(t) and s(t) can have **independent numbers of control points**:

```json
"optimization_parameters": {
  "max_evaluations": 10000,
  "control_times": [0, 100, 200, 400],      // 4 points for f
  "initial_guess": [0.5, 0.4, 0.5, 0.6],
  "s_control_times": [0, 200, 400],         // 3 points for s
  "s_initial_guess": [0.24, 0.26, 0.22],
  "algorithm": "LN_SBPLX",
  "xtol_abs": 1e-10
}
```

**Key features:**
- **Independent control times:** f and s can have different temporal resolution
- **Independent initial guesses:** Different starting points for each variable
- **Interpolation:** Both variables use PCHIP interpolation between control points
- **Total dimension:** n_f + n_s (e.g., 4 + 3 = 7 dimensions in example above)

#### Backward Compatibility

If `s_control_function` is **not** present in the configuration:
- Single-variable optimization mode (f only)
- s(t) comes from `time_functions['s']` and is fixed during optimization
- All existing configurations continue to work without modification

#### Example: Testing Different s(t) Trajectories

To test prescribed s(t) trajectories without optimization:

```json
{
  "time_functions": {
    "s": {
      "type": "piecewise_linear",
      "time_points": [0, 400],
      "values": [0.30, 0.20]
    }
  },
  "control_function": {
    "type": "constant",
    "value": 0.5
  }
}
```

This runs the model with f=0.5 and s declining linearly from 0.30 to 0.20, without invoking dual optimization.

## Next Steps

The following tasks are prioritized to prepare the model for production use and publication:

### 1. Debug Model for Income Redistribution Cases

Verify and debug the model to ensure it is working correctly for income redistribution scenarios:
- Test model behavior across range of inequality levels (Gini coefficients)
- Validate income redistribution mechanics and effective Gini calculations
- Check that climate damage with income-dependent effects produces physically reasonable results
- Verify that optimization converges to sensible policies for redistribution vs. abatement tradeoff
- Document any issues discovered and ensure all calculations are correct

### 2. Explore Model Sensitivities

Systematically explore how model results depend on key parameters and assumptions:
- Parameter sensitivity analysis (discount rate, risk aversion, damage functions, etc.)
- Sensitivity to initial conditions (initial capital, cumulative emissions, inequality)
- Sensitivity to time-dependent function specifications (TFP growth, population, carbon intensity)
- Document parameter ranges that produce realistic and stable model behavior
- Identify which parameters most strongly influence optimal policies

### 3. Design and Execute Production Simulations

Design and run the final simulation suite for publication:
- Define baseline and alternative scenarios to be presented in the paper
- Create production configuration files for each scenario
- Run optimizations to determine optimal policies
- Generate publication-quality figures and tables
- Document key results and policy insights
- Ensure reproducibility of all results

## Dual Optimization of Savings Rate and Abatement Allocation

### Overview

The model now supports simultaneous optimization of both:
1. **f(t)** - allocation fraction between abatement and redistribution
2. **s(t)** - savings rate (fraction of net output invested)

This capability allows the model to optimize the tradeoff between present Consumption and future Consumption (via savings/investment) while simultaneously optimizing the allocation of resources between climate mitigation and inequality reduction.

**Implementation Status:** ✅ Infrastructure complete (Phases 1-5 done)
- Dual control functions operational
- Independent control points for f and s
- Full backward compatibility maintained
- Economic behavior validated

**Remaining Work:** Phase 6 (documentation and visualization), plus automated dual optimization method (future work)

### Implementation Plan

The implementation will parallel the existing optimization structure for f(t), creating independent control points and decision times for s(t):

#### Phase 1: Make Savings Rate Time-Dependent
**Status:** ✅ Completed

1. **Move s from scalar_params to time_functions** ✅
   - Removed `s` from `ScalarParameters` dataclass in `parameters.py`
   - Added `s` to time-dependent functions in configuration files
   - Updated `evaluate_params_at_time()` to evaluate `s(t)` from time functions
   - s(t) is evaluated at each timestep and can be bounded by configuration

2. **Update all code references to s** ✅
   - In `economic_model.py`: s is now treated as time-dependent parameter from `params` dict
   - In `integrate_model()`: s is obtained from `evaluate_params_at_time(t_start, config)`
   - Initial capital iteration correctly uses time-dependent s(0)
   - Verified s is correctly stored in results and output to CSV

3. **Update configuration files** ✅
   - Converted `s` from scalar parameter to time function in both config files:
     - `config_test_DICE.json`
     - `config_test_DICE_2x_0.02_10k_0.67.json`
   - Using `"type": "constant"` with same value (0.23974) to preserve behavior
   - Tested: time-dependent s(t) = constant reproduces previous results exactly

4. **Documentation updates** ✅
   - Updated `test_integration.py` to display s(0) from time_functions
   - Docstrings in `evaluate_params_at_time()` updated to list s as time-dependent
   - s is now part of the time-dependent evaluation pipeline alongside A, L, sigma, theta1

#### Phase 2: Extend Control Function Structure
**Status:** ✅ Completed

1. **Modify control function to return both f and s** ✅
   - Created `create_dual_control_from_single(f_control, s_time_function)` to wrap f control and s time function
   - Created `create_dual_control_from_specs(f_spec, s_spec)` for future dual control specifications
   - Control function now returns tuple `(f, s)` instead of scalar

2. **Update ModelConfiguration** ✅
   - `control_function` now returns `(f(t), s(t))` tuple
   - In `load_configuration()`, f control is wrapped with s time function to create dual control
   - Clean transition: f comes from control_function spec, s from time_functions

3. **Update evaluate_params_at_time()** ✅
   - Unpacks tuple from control_function: `f, s = config.control_function(t)`
   - Both f and s are added to params dict for use by economic model
   - Removed direct s evaluation from time_functions (now comes from control function)

4. **Verified integrate_model() compatibility** ✅
   - No changes needed in integrate_model() - all control function calls go through evaluate_params_at_time()
   - Both f and s correctly stored in results and CSV output
   - Backward compatibility maintained: constant s(t) produces identical results

#### Phase 3: Extend Optimization Framework
**Status:** ✅ Completed (Infrastructure)

1. **Created dual control function builder** ✅
   - Added `create_dual_control_function_from_points(f_control_points, s_control_points)` in optimization.py
   - Interpolates f and s independently using separate control point sets
   - Returns callable function: `(f, s) = control_function(t)`
   - Supports different numbers of control points and time spacing for each variable

2. **Extended calculate_objective for dual control** ✅
   - Added optional parameters: `s_control_values` and `s_control_times`
   - **Backward compatible**: If s parameters omitted, uses fixed s from time_functions
   - **Dual mode**: If s parameters provided, optimizes both f and s independently
   - Both control values clamped to [0, 1] automatically

3. **Verified backward compatibility** ✅
   - Tested: Single-variable optimization (f only) works identically to before
   - s remains fixed from configuration when not explicitly optimized
   - Existing optimization workflows unchanged
   - Objective function calculation: 3.28e13 (verified)

4. **Infrastructure ready for full dual optimization**
   - Foundation in place for extending `optimize_control_function()` to handle dual parameter vectors
   - Can manually optimize both f and s by calling `calculate_objective()` with full parameter sets
   - Future work: Integrate into `optimize_control_function()` and iterative refinement for automated dual optimization

**Current capabilities:**
- ✅ Optimize f while keeping s fixed (default, backward compatible)
- ✅ Infrastructure supports dual optimization via direct calls to `calculate_objective()`
- ⏳ Automated dual optimization via `optimize_control_function()` (Phase 4+)

#### Phase 4: Configuration and Initial Guesses
**Status:** ✅ Completed

1. **Updated configuration file structure** ✅
   - Added optional `s_control_function` field parallel to `control_function`
   - If `s_control_function` present: s is treated as control variable
   - If absent: s comes from `time_functions['s']` (backward compatible)
   - Clean precedence: s_control_function > time_functions['s']

2. **Extended OptimizationParameters dataclass** ✅
   - Added `s_control_times`: int or list, parallel to `control_times` for f
   - Added `s_initial_guess`: float or list, parallel to `initial_guess` for f
   - Added `s_n_points_final`: int, for iterative refinement of s
   - Added `is_dual_optimization()` method to check if optimizing both f and s
   - All fields optional with None default (backward compatible)

3. **Updated load_configuration()** ✅
   - Checks for optional `s_control_function` in JSON
   - If present: creates dual control from both f and s control functions
   - If absent: creates dual control from f control + s time function (default)
   - Uses `create_dual_control_from_specs()` for dual control mode

4. **Created test configurations** ✅
   - Created `config_test_dual_simple.json` with both f and s as controls
   - Tested with constant values: f=0.5, s=0.24
   - Verified integration works correctly
   - Confirmed CSV output shows s from s_control_function (0.24) not time_functions (0.23974)

**Current capabilities:**
- ✅ Configuration files can specify dual optimization
- ✅ OptimizationParameters track both f and s control specifications
- ✅ Full backward compatibility when s_control_function absent
- ✅ Infrastructure ready for automated dual optimization in future phases

#### Phase 5: Testing and Validation
**Status:** ✅ Completed

1. **Verify backward compatibility** ✅
   - ✅ With s(t) = constant, results match previous version exactly (objective 3.28e13)
   - ✅ All existing configurations run correctly

2. **Test time-dependent s** ✅
   - ✅ Linear ramp s: 0.20→0.30 trajectory verified
   - ✅ Economic variables respond correctly to changing s(t)
   - ✅ Capital accumulation dynamics work as expected

3. **Test dual optimization infrastructure** ✅
   - ✅ Single control point for both (2D optimization via grid search)
   - ✅ Multiple control points with same number (n_f = n_s = 3)
   - ✅ Multiple control points with different numbers (n_f = 2, n_s = 4 and vice versa)
   - ⚠️ Iterative refinement not yet implemented (deferred to future work)

4. **Verify economic sensibility** ✅
   - ✅ Different s(t) trajectories produce different outcomes (0.51% variation)
   - ✅ Best constant s = 0.30 in typical range [0.20, 0.35]
   - ✅ Higher savings improves utility by 0.17% over baseline
   - ✅ Declining s trajectory outperforms increasing s (matches economic intuition)

**Key accomplishments:**
- Dual control infrastructure fully functional
- Independent control points for f and s verified
- Economic behavior consistent with growth theory
- Full backward compatibility maintained

#### Phase 6: Documentation and Output
**Status:** ✅ Completed

1. **Update README.md** ✅
   - ✅ Documented dual control variable framework in "Control Variables" section
   - ✅ Added "Dual Optimization (f and s)" section in "Optimization Configuration"
   - ✅ Explained s(t) optimization and configuration options
   - ✅ Added examples of dual optimization configurations (basic and multi-point)
   - ✅ Documented backward compatibility behavior
   - ✅ Showed example of prescribed s(t) trajectory testing

2. **Update output visualization** ✅
   - ✅ s(t) plotted alongside f(t) in "Control Variables" combined chart
   - ✅ Both variables shown in dimensionless_ratios plot group
   - ✅ Visualization automatically handles both single and dual optimization modes
   - ✅ Uses existing combined chart infrastructure for clean f/s comparison

3. **Update optimization summary output** ✅
   - ✅ write_optimization_summary() reports optimal f(t) control points
   - ✅ write_optimization_summary() reports optimal s(t) control points (when present)
   - ✅ Separate sections for f and s control point tables
   - ✅ Infrastructure ready for full dual optimization method (future work)

**Key accomplishments:**
- Complete documentation of dual control framework
- User-facing examples for all dual optimization modes
- Output visualization supports both f and s
- Optimization summary ready for dual optimization results

### Key Design Decisions

1. **Independent control times**: Each variable (f, s) has its own set of control times, allowing different temporal resolution where needed

2. **Parallel iteration**: During iterative refinement, both variables go through the same number of iterations, but may have different numbers of control points

3. **Minimal structural change**: The implementation leverages the existing optimization framework, simply extending the parameter vector dimension

4. **Backward compatibility**: Configurations can specify constant s(t) to reproduce current behavior exactly

### Expected Benefits

- **More realistic optimization**: Savings rate is a key economic policy variable that should be optimized alongside other controls
- **Richer dynamics**: Time-varying s(t) allows model to balance present vs. future Consumption optimally
- **Methodological advancement**: Demonstrates framework extensibility to multi-dimensional control problems

## Project Structure

```
coin_equality/
├── README.md                          # This file
├── CLAUDE.md                          # AI coding style guide
├── requirements.txt                   # Python dependencies
├── income_distribution.py             # Core income distribution functions
├── economic_model.py                  # Economic production and tendency calculations
├── parameters.py                      # Parameter definitions and configuration loading
├── output.py                          # Output generation (CSV and PDF)
├── test_integration.py                # Test script demonstrating complete workflow
├── config_baseline.json               # Baseline scenario configuration
├── config_high_inequality.json        # High inequality scenario configuration
├── data/output/                       # Output directory (timestamped subdirectories)
├── coin_equality (methods) v0.1.pdf   # Detailed methods document
└── [source code directories]
```

## References

Barrage, L., & Nordhaus, W. (2024). "Policies, projections, and the social cost of carbon: Results from the DICE-2023 model." *Proceedings of the National Academy of Sciences*, 121(13), e2312030121. https://doi.org/10.1073/pnas.2312030121

Caldeira, K., Bala, G., & Cao, L. (2023). "Climate sensitivity uncertainty and the need for energy without CO₂ emission." *Environmental Research Letters*, 18(9), 094021. https://doi.org/10.1088/1748-9326/acf949

Nordhaus, W. D. (1992). "An optimal transition path for controlling greenhouse gases." *Science*, 258(5086), 1315-1319. https://doi.org/10.1126/science.258.5086.1315

Nordhaus, W. D. (2017). "Revisiting the social cost of carbon." *Proceedings of the National Academy of Sciences*, 114(7), 1518-1523. https://doi.org/10.1073/pnas.1609244114

## License

MIT License

Copyright (c) 2025 Lamprini Papargyri, ..., and Ken Caldeira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Authors

Lamprini Papargyri, ..., and Ken Caldeira
