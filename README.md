# COIN_equality

A simple-as-possible stylized representation of the tradeoff between investment in income redistribution versus investment in emissions abatement.

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
4. **Ω, G_climate** from ΔT, Gini, y_gross, damage params (Eq 1.2: income-dependent climate damage and distributional effect; when ΔL >= 1, uses uniform damage approximation)
5. **Y_damaged** from Y_gross, Ω (Eq 1.3: production after climate damage)
6. **y** from Y_damaged, L, s (Eq 1.4: mean per-capita income after climate damage)
7. **Δc** from y, ΔL (Eq 4.3: per-capita amount redistributable)
8. **E_pot** from σ, Y_gross (Eq 2.1: potential emissions)
9. **abatecost** from f, Δc, L (Eq 1.5: abatement expenditure)
10. **μ** from abatecost, θ₁, θ₂, E_pot (Eq 1.6: fraction of emissions abated, capped at μ_max)
11. **Λ** from abatecost, Y_damaged (Eq 1.7: abatement cost fraction)
12. **Y_net** from Y_damaged, Λ (Eq 1.8: production after abatement costs)
13. **y_eff** from y, abatecost, L (Eq 1.9: effective per-capita income)
14. **G_eff** from f, ΔL, G_climate (Eq 4.4: effective Gini after redistribution/abatement; when ΔL >= 1, G_eff = G_climate with no redistribution effect)
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
All integrals are solved analytically using closed-form solutions based on hypergeometric functions. This avoids numerical integration and is exact within numerical precision. See `climate_damage_distribution.py` for complete derivations and implementation.

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
abatecost(t) = f · Δc(t) · L(t)
```
This is the total amount society allocates to emissions abatement, where:
- `f` = fraction of redistributable resources allocated to abatement (0 ≤ f ≤ 1)
- `Δc(t)` = per-capita amount of income available for redistribution
- `L(t)` = population

**Eq. (1.6) - Abatement Fraction:**
```
μ(t) = min(μ_max, [abatecost(t) · θ₂ / (E_pot(t) · θ₁(t))]^(1/θ₂))
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
Λ(t) = abatecost(t) / Y_damaged(t)
```
This represents the fraction of damaged production allocated to emissions abatement.

**Eq. (1.8) - Net Production:**
```
Y_net(t) = (1 - Λ(t)) · Y_damaged(t)
```
Production after both climate damage and abatement costs.

**Eq. (1.9) - Effective Per-Capita Income:**
```
y_eff(t) = y(t) - abatecost(t) / L(t)
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
- `ΔL` = fraction of total income to be redistributed (specified exogenously)
- `f` = fraction of redistributable resources allocated to abatement (0 ≤ f ≤ 1)

**Eq. (4.1) - Fraction of Income Redistributed:**

Given `ΔL` and `G₁`, we numerically solve for `G₂` (the Gini index after full redistribution) using the relationship:
```
ΔL(F*) = [2(G₁-G₂)/(1-G₁)(1+G₂)] · [((1+G₁)(1-G₂))/((1-G₁)(1+G₂))]^((1+G₁)(1-G₂)/(2(G₂-G₁)))
```
where `F*` is the crossing rank (see Eq. 4.2).

**Eq. (4.2) - Crossing Rank:**

The population rank where income remains unchanged during redistribution:
```
F* = 1 - [((1+G₁)(1-G₂))/((1-G₁)(1+G₂))]^(((1+G₁)(1+G₂))/(2(G₂-G₁)))
```

**Eq. (4.3) - Per-Capita Amount Redistributed:**
```
Δc = y · ΔL
```
where `y` is mean per-capita income.

**Eq. (4.4) - Effective Gini Index:**

When fraction `f` of redistributable resources goes to abatement instead of redistribution, the effective Gini index is calculated using a two-step Pareto-preserving approach (see `income_distribution.calculate_Gini_effective_redistribute_abate`).

For reference, the formula is:
```
G_eff(f) = (1-ΔL)/(1-f·ΔL) · [1 - (1 - G₁)^((1-ΔL(1-F*))/(1-ΔL))]
```

where:
- `f = 0`: all resources go to redistribution → `G_eff(0)` = minimum (most equal)
- `f = 1`: all resources go to abatement → `G_eff(1)` = maximum Gini given removal
- `0 < f < 1`: mixed allocation

**Fraction of Emissions Abated:**

See Eq. (1.6) above. The abatement fraction is determined by the amount society allocates to abatement relative to potential emissions and the marginal abatement cost.

**Enhanced Redistribution Mode (ΔL >= 1) - Partial Implementation**

The model now supports `ΔL >= 1` with special handling that disables redistribution and allows pure abatement optimization.

**Current Behavior (ΔL < 1):**
- Redistribution operates within the Pareto family of distributions
- Income transfers preserve the general shape of the distribution
- The Gini coefficient changes, but the underlying distribution remains Pareto
- Climate damage calculations assume a fixed Pareto distribution with parameter `a` derived from current Gini
- Control variable `f` determines allocation between abatement and redistribution

**Implemented Behavior (ΔL >= 1):**
When `ΔL >= 1`, the model disables redistribution and enables pure abatement optimization:

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
   - Available budget: `delta_c = y * delta_L` (Line 136 of `economic_model.py`)
   - With `delta_L >= 1`, this creates `delta_c >= y` (budget at least equals full per-capita income)
   - Abatement expenditure: `abatecost = f * delta_c * L` (Line 142)
   - Effective income: `y_eff = y - abatecost/L = y - f * delta_c`

4. **Optimizer Behavior**:
   - The optimizer chooses `f` to maximize utility over time
   - **Naturally selects `f << 1`** because:
     - Large `f` would make `y_eff = y - f * delta_c` very small or negative
     - This would result in terrible current utility (consumption crash)
     - Optimizer balances current consumption vs. future climate benefits
   - **Equivalence**: Optimization of `f` becomes equivalent to optimizing the abatement/consumption tradeoff
   - No redistribution component in utility calculation (since `G_eff = Gini_climate`)

5. **Physical Interpretation**:
   - `ΔL >= 1` represents a model mode where redistribution is turned off
   - Allows studying pure abatement policy without redistribution considerations
   - Budget parameter `ΔL` scales the available resources, but optimizer self-limits via utility constraints
   - Climate damage treated as uniform across income levels (first-order approximation)

**Implementation Status**:
- ✓ Redistribution disabled in `economic_model.py` (lines 164-167)
- ✓ Uniform damage approximation in `climate_damage_distribution.py` (line 156)
- ✓ Uses `INVERSE_EPSILON` constant from `constants.py` (no hardcoded values)
- ✓ All existing unit tests pass

**Future Enhancements**:
For more sophisticated treatment of `ΔL >= 1` with non-Pareto income distributions, see **Next Steps, Section 2**.

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
| `ΔL` | Fraction of income available for redistribution (<1: active redistribution; >=1: redistribution disabled, pure abatement mode) | - | `deltaL` |

### Time-Dependent Functions

These functions are evaluated at each time step:

| Function | Description | Units | JSON Key |
|----------|-------------|-------|----------|
| `A(t)` | Total factor productivity | - | `A` |
| `L(t)` | Population | people | `L` |
| `σ(t)` | Carbon intensity of GDP | tCO₂ $⁻¹ | `sigma` |
| `θ₁(t)` | Marginal abatement cost as μ→1 | $ tCO₂⁻¹ | `theta1` |

Each function is specified by `type` and type-specific parameters (e.g., `initial_value`, `growth_rate`). Six function types are available: `constant`, `exponential_growth`, `logistic_growth`, `piecewise_linear`, `double_exponential_growth` (Barrage & Nordhaus 2023), and `gompertz_growth` (Barrage & Nordhaus 2023). See the Configuration section below for detailed specifications.

### Control Variable

| Variable | Description | Units | JSON Key |
|----------|-------------|-------|----------|
| `f(t)` | Fraction of redistributable resources allocated to abatement | - | `control_function` |

The control function determines the allocation between emissions abatement and income redistribution (0 = all to redistribution, 1 = all to abatement).

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

### Inverse Problem: Finding G2 from ΔL

- **`_phi(r)`** - Helper function for numerical root finding; computes `φ(r) = (r-1) · r^(1/(r-1)-1)` with proper handling of edge cases

- **`G2_from_deltaL(deltaL, Gini_initial)`** - **Solves the inverse problem**: given an initial Gini `Gini_initial` and a desired redistribution amount `ΔL`, numerically finds the target Gini `G2` that would result from full redistribution. Uses `scipy.optimize.root_scalar` with Brent's method. Returns `(G2, remainder)` where remainder is non-zero if `ΔL` exceeds the maximum possible for the Pareto family (caps at G2=0).

### Effective Gini Calculation

- **`calculate_Gini_effective_redistribute_abate(f, deltaL, Gini_initial)`** - **Main function** that calculates the effective Gini index when fraction `f` of redistributable resources is allocated to emissions abatement instead of redistribution.

  **Algorithm:**
  1. Solves for full-redistribution target `G2` from `ΔL` and `Gini_initial`
  2. Computes crossing rank `F*` for the `(Gini_initial → G2)` transition
  3. Calculates effective redistribution amount `ΔL_eff` at the same `F*` for partial allocation
  4. Solves for Pareto-equivalent `G2_eff` from `ΔL_eff`

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
   - Distribution: `Gini_initial`, `Gini_fract`, `Gini_restore`, `deltaL`

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
- **`K(0)`**: Steady-state capital stock with no climate damage or abatement:
  ```
  K₀ = (s · A(0) / δ)^(1/(1-α)) · L(0)
  ```
- **`Gini(0) = Gini_initial`**: Initial Gini index from configuration

This ensures the model starts from a consistent economic equilibrium. Setting `Ecum_initial` > 0 allows modeling scenarios with pre-existing climate change (e.g., starting from year 2020 conditions).

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

- Uses `mpmath` library for arbitrary-precision arithmetic (80 decimal places)
- Numerical integration performed with `mpmath.quad()` adaptive quadrature
- Tests both the aggregate damage (Ω) and implicitly validates the underlying income distribution formulas
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

### Output Variables

The results dictionary contains arrays for:
- **Time**: `t`
- **State variables**: `K`, `Ecum`, `Gini`
- **Time-dependent inputs**: `A`, `L`, `sigma`, `theta1`, `f`
- **Economic variables**: `Y_gross`, `Y_damaged`, `Y_net`, `y`, `y_eff`
- **Climate variables**: `delta_T`, `Omega`, `E`
- **Abatement variables**: `mu`, `Lambda`, `abatecost`, `delta_c`
- **Inequality/utility**: `G_eff`, `U`
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
- Time column ('t') is always the first column, followed by other variables in alphabetical order
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
2. **Effective consumption discount rate**: `r_c = ρ + η · k_A · (1 - α)`
   - `ρ`: pure rate of time preference
   - `η`: coefficient of relative risk aversion
   - `α`: capital share of income

For iteration with N+1 control points (k = 0, 1, ..., N):
```
t(k) = -(1/r_c) · ln(1 - (k/N) · (1 - exp(-r_c · t_end)))
```

This formula ensures that each interval between control points contributes approximately equally to the discounted objective, with more points concentrated in early periods where discounting matters most.

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

## Next Steps

The following tasks are prioritized to prepare the model for production use and publication:

### 1. Align Model Components with DICE-2023 (Barrage & Nordhaus 2024)

Update key model components to more closely follow the formulations in Barrage and Nordhaus (2024):

**Climate damage function:**
- ✓ Complete: Now uses DICE-2023 formulation: `ω_max(ΔT) = psi1 · ΔT + psi2 · ΔT²` [Barrage & Nordhaus 2023]

**Carbon intensity (σ):**
- Current: Simple exponential decay
- Target: DICE-2023 carbon intensity trajectory with technological progress

**Backstop price (θ₁):**
- Current: Simple exponential decline in marginal abatement cost
- Target: DICE-2023 backstop price formulation with cost reductions

**Total factor productivity (A):**
- Current: Simple exponential growth
- Target: DICE-2023 TFP trajectory with calibrated growth rates

This alignment will ensure our extensions (income distribution, redistribution mechanisms) are built on a well-established baseline that matches current IAM best practices.

### 2. Enhanced Redistribution Mode (ΔL >= 1) - Advanced Features

**Status**: ✓ Basic implementation complete. Redistribution is disabled and uniform damage approximation is used when `ΔL >= 1`.

**Current Implementation**:
- Redistribution turned off: `G_eff = Gini_climate` (no redistribution effect)
- Uniform damage: `Omega = omega_max`, `Gini_climate = Gini_current`
- Optimizer naturally selects `f << 1` due to utility constraints
- Allows studying pure abatement policy without redistribution

**Future Advanced Features** (optional enhancements for more sophisticated treatment):

If desired to model actual structural redistribution with non-Pareto distributions:

1. **Implement non-uniform climate damage with non-Pareto distributions**:
   - Replace uniform damage approximation with income-dependent calculation
   - Determine appropriate functional form for income distribution when departing from Pareto
   - Either extend analytical solutions (hypergeometric functions) or implement numerical integration
   - Validate that damage vulnerability profile remains physically reasonable

2. **Model actual structural redistribution**:
   - Define how `ΔL >= 1` maps to distribution shape changes
   - Implement redistribution mechanics beyond Pareto family in `income_distribution.py`
   - Update `economic_model.py` to calculate modified distribution parameters
   - Ensure continuous transition at `ΔL = 1` boundary

3. **Testing and validation**:
   - Unit tests for boundary behavior at `ΔL = 1`
   - Asymptotic tests as `ΔL` increases
   - Comparison against analytical solutions where available

**Key Design Questions**:
- What functional form for income distribution when departing from Pareto?
- How should climate damage depend on income in non-Pareto distributions?
- What are physically reasonable upper bounds on `ΔL`?
- Is the added complexity justified for modeling purposes?

**Note**: Current implementation (redistribution disabled, uniform damage) may be sufficient for most use cases. These enhancements are optional and should only be pursued if needed for specific research questions.

### 3. Update Methods Section of Paper

Revise and update the Methods section of the paper to ensure it accurately reflects the current implementation as documented in this README and the model code. The paper should provide a clear, consistent description of all model equations, parameter definitions, and computational approaches used in the codebase.

### 4. Comprehensive Code Validation

Perform a detailed verification of model calculations by manually tracing through one complete time step of the integration:
- Use the output `results.csv` file to verify intermediate calculations
- Check that all equations are implemented correctly and consistently with documentation
- Validate state variable updates, tendency calculations, and functional relationships
- Ensure numerical values propagate correctly through the calculation order
- Document any discrepancies or unexpected behaviors

This step-by-step verification will provide confidence in the correctness of the implementation.

### 5. Mathematica Verification of Model Equations

Use Wolfram Mathematica to independently re-derive and verify all model equations:

**Analytical derivations to verify:**
- Pareto-Lorenz income distribution and Gini coefficient relationships
- Mean utility calculation with income distribution (Eq. 3.5)
- Income redistribution mechanics and effective Gini formulation (Eq. 4.4)
- Climate damage with income-dependent effects and distributional impacts
- Abatement cost functions and emission relationships
- All closed-form solutions and integrals

**Numerical verification:**
- Compare Mathematica symbolic solutions with Python numerical implementations
- Verify hypergeometric function evaluations in climate damage calculations
- Check integration of utility across income distribution
- Validate PCHIP interpolation behavior at boundaries

**Benefits:**
- Independent verification ensures mathematical correctness
- Symbolic computation catches algebraic errors that may not appear in numerical tests
- Provides publication-ready analytical expressions
- Validates assumptions in closed-form approximations
- Ensures consistency between documentation and implementation

This verification step provides confidence that the model mathematics is sound before using it for policy analysis.

### 6. Production Code Readiness

Establish confidence that the model is ready for production use:
- Confirming all calculations are correct and well-tested
- Ensuring optimization routines reliably find optimal solutions
- Documenting any known limitations or edge cases
- Creating comprehensive test cases that verify expected model behavior
- Establishing this codebase as a reliable tool for research and analysis

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
