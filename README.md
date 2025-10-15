# COIN_equality

A simple-as-possible stylized representation of the tradeoff between investment in income redistribution versus investment in emissions abatement.

## Overview

This project develops a highly stylized model of an economy with income inequality, where a specified fraction of gross production is allocated to social good. The central question is how to optimally allocate resources between two competing objectives:

1. **Income redistribution** - reducing inequality by transferring income from high-income to low-income individuals
2. **Emissions abatement** - reducing carbon emissions to mitigate future climate damage

The model extends the COIN framework presented in [Caldeira et al. (2023)](https://doi.org/10.1038/s41586-023-06017-3) to incorporate income inequality and diminishing marginal utility of income.

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

1. **Y_gross** from K, L, A, α (Cobb-Douglas production)
2. **ΔT** from Ecum, k_climate (temperature from cumulative emissions)
3. **Ω** from ΔT, k_damage, β (climate damage fraction)
4. **Y_net** from Y_gross, Ω (production after climate damage)
5. **y** from Y_net, L, s (mean per-capita income)
6. **Δc** from y, ΔL (per-capita amount redistributable)
7. **μ** from f, Δc, θ₁, θ₂ (fraction of emissions abated)
8. **E** from σ, μ, Y_gross (emissions net of abatement)
9. **dK/dt** from s, Y_net, δ, K (capital tendency)
10. **dEcum/dt = E** (cumulative emissions tendency)

Additionally, for utility calculations:
- **G_eff** from f, ΔL, G₁ (effective Gini index)
- **U** from y_eff, G_eff, η (mean utility)

### Core Components

#### 1. Economic Model (Solow-Swann Growth)

**Eq. (1.1) - Production Function (Cobb-Douglas):**
```
Y_gross(t) = A(t) · K(t)^α · L(t)^(1-α)
```

**Eq. (1.2) - Climate Damage:**
```
Ω(t) = k_damage · ΔT(t)^β
```
where `Ω(t)` is the fraction of gross production lost to climate damage.

**Eq. (1.3) - Net Production (after climate damage):**
```
Y_net(t) = (1 - Ω(t)) · Y_gross(t)
```

**Eq. (1.4) - Abatement Fraction:**

The fraction of emissions abated, `μ(t)`, is determined by the allocation between redistribution and abatement:
```
μ(t) = [f·Δc(t)·L(t) / (θ₁(t)·L(t))]^(1/θ₂)
     = [f·Δc(t) / θ₁(t)]^(1/θ₂)
```
where:
- `Δc(t)` = per-capita amount of income that could be redistributed
- `f` = fraction of redistributable resources allocated to abatement (0 ≤ f ≤ 1)
- `θ₁(t)` = abatement cost coefficient
- `θ₂` = abatement cost exponent

**Eq. (1.5) - Abatement Cost Fraction:**
```
Λ(t) = θ₁(t) · μ(t)^θ₂
```
This represents the fraction of gross production allocated to emissions abatement.

**Eq. (1.6) - Abatement Cost:**
```
abatecost(t) = Λ(t) · Y_net(t)
```

**Eq. (1.7) - Mean Per-Capita Income:**
```
y(t) = (1 - s) · Y_net(t) / L(t)
```

**Eq. (1.8) - Effective Per-Capita Income:**
```
y_eff(t) = y(t) - abatecost(t) / L(t)
```
This is the per-capita income after subtracting abatement costs, used for utility calculations.

**Eq. (1.9) - Capital Accumulation:**
```
dK/dt = s · Y_net(t) - δ · K(t)
```

#### 2. Climate Model

**Eq. (2.1) - Emissions:**
```
E_base(t) = σ(t) · Y_gross(t)
E(t) = σ(t) · (1 - μ(t)) · Y_gross(t)
```

**Eq. (2.2) - Temperature Change:**
```
ΔT(t) = k_climate · ∫₀^t E(t') dt'
       = k_climate · Ecum(t)
```

Temperature change is proportional to cumulative carbon dioxide emissions.

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

When fraction `f` of redistributable resources goes to abatement instead of redistribution, the effective Gini index is calculated using a two-step Pareto-preserving approach (see `income_distribution.G2_effective_pareto`).

For reference, the formulas are:
```
G₂ᵣ = 1 - (1 - G₁)^((1-ΔL(1-F*))/(1-ΔL))  [all to abatement, f=1]

G_eff(f) = (1-ΔL)/(1-f·ΔL) · [1 - (1 - G₁)^((1-ΔL(1-F*))/(1-ΔL))]
```

where:
- `f = 0`: all resources go to redistribution → `G_eff(0)` = minimum (most equal)
- `f = 1`: all resources go to abatement → `G_eff(1) = G₂ᵣ`
- `0 < f < 1`: mixed allocation

**Fraction of Emissions Abated:**
```
μ(t) = [f·Δc(t)·L(t) / (θ₁(t)·L(t))]^(1/θ₂)
```

## Key Parameters

| Symbol | Description | Units |
|--------|-------------|-------|
| `ρ` | Pure rate of time preference | yr⁻¹ |
| `η` | Coefficient of relative risk aversion | - |
| `G` | Gini index (0 = perfect equality, 1 = maximum inequality) | - |
| `α` | Output elasticity of capital (capital share of income) | - |
| `s` | Savings rate | - |
| `δ` | Capital depreciation rate | yr⁻¹ |
| `k_damage` | Climate damage coefficient | °C⁻ᵝ |
| `β` | Climate damage exponent | - |
| `σ(t)` | Carbon intensity of GDP | tCO₂ $⁻¹ |
| `θ₁(t)` | Abatement cost coefficient | - |
| `θ₂` | Abatement cost exponent | - |
| `k_climate` | Temperature sensitivity to cumulative emissions | °C tCO₂⁻¹ |
| `f` | Fraction of redistributable resources allocated to abatement | - |

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

- **`crossing_rank_from_G(G1, G2)`** - Computes the population rank `F*` where income remains unchanged during redistribution from `G1` to `G2` (equation 10)

- **`deltaL_G1_G2(G1, G2)`** - Calculates the fraction of total income `ΔL` redistributed when Gini shifts from `G1` to `G2` (equation 11). Uses log-space arithmetic for numerical stability.

### Inverse Problem: Finding G2 from ΔL

- **`_phi(r)`** - Helper function for numerical root finding; computes `φ(r) = (r-1) · r^(1/(r-1)-1)` with proper handling of edge cases

- **`G2_from_deltaL(deltaL, G1)`** - **Solves the inverse problem**: given an initial Gini `G1` and a desired redistribution amount `ΔL`, numerically finds the target Gini `G2` that would result from full redistribution. Uses `scipy.optimize.root_scalar` with Brent's method. Returns `(G2, remainder)` where remainder is non-zero if `ΔL` exceeds the maximum possible for the Pareto family (caps at G2=0).

### Effective Gini Calculation

- **`G2_effective_pareto(f, deltaL, G1)`** - **Main function** that calculates the effective Gini index when fraction `f` of redistributable resources is allocated to emissions abatement instead of redistribution.

  **Algorithm:**
  1. Solves for full-redistribution target `G2` from `ΔL` and `G1`
  2. Computes crossing rank `F*` for the `(G1 → G2)` transition
  3. Calculates effective redistribution amount `ΔL_eff` at the same `F*` for partial allocation
  4. Solves for Pareto-equivalent `G2_eff` from `ΔL_eff`

  **Parameters:**
  - `f = 0`: All resources to redistribution → minimum Gini (maximum equality)
  - `f = 1`: All resources to abatement → maximum Gini given removal
  - `0 < f < 1`: Mixed allocation

  **Returns:** `(G2_eff, remainder)` tuple

### Usage Example

```python
from income_distribution import G2_effective_pareto

# Initial Gini index
G1 = 0.4

# Fraction of income to be redistributed (e.g., 5% of total income)
deltaL = 0.05

# Fraction allocated to abatement vs redistribution
f = 0.5  # 50% to abatement, 50% to redistribution

# Calculate effective Gini index
G_eff, remainder = G2_effective_pareto(f, deltaL, G1)

print(f"Effective Gini: {G_eff:.4f}")
```

## Next Steps

### 1. Define Test Case Parameters and Exogenous Functions

Create a test configuration with:
- Initial conditions: `K(0)`, `Ecum(0)`
- Constant parameters: `α`, `δ`, `s`, `k_damage`, `β`, `k_climate`, `η`, `G₁`, `ΔL`
- Time-dependent exogenous functions: `A(t)`, `L(t)`, `σ(t)`, `θ₁(t)`, `θ₂`

These should be defined as Python functions or data structures that can be evaluated at any time `t`.

### 2. Create Time-Integration Routine

Develop a forward model that integrates the system from `t=0` to `t=T` given:
- Initial state: `state_0 = {'K': K0, 'Ecum': Ecum0}`
- Control trajectory: `f(t)` - the fraction allocated to abatement vs redistribution
- Parameters: including time-dependent functions

Use `scipy.integrate.solve_ivp` or similar ODE solver. The control variable `f(t)` will later be optimized but initially should be tested with simple functions (e.g., constant, linear, step functions).

### 3. Test Forward Model

Validate the forward integration:
- Run with physically reasonable parameters
- Check conservation properties and bounds (0 ≤ f(t) ≤ 1, K > 0, etc.)
- Verify sensitivity to key parameters
- Compare different `f(t)` trajectories manually
- Calculate and track utility over time using `y_eff(t)` and `G_eff(t)`

### 4. Create Optimization Code

Find the optimal control trajectory `f(t)` that maximizes the objective function:

```
max ∫₀^T e^(-ρt) · U(t) · L(t) dt
```

where `U(t)` is calculated from `y_eff(t)`, `G_eff(t)`, and `η` (Eq. 3.5).

**Optimization approaches to consider:**

1. **MIDACO Solver** (used previously): https://www.midaco-solver.com/
   - Commercial global optimization solver
   - Handles mixed-integer nonlinear problems
   - Good for constrained optimization

2. **scipy.optimize alternatives:**
   - `scipy.optimize.minimize` - local optimization with constraints
   - `scipy.optimize.differential_evolution` - global optimization
   - `scipy.optimize.dual_annealing` - global optimization

3. **Optimal control solvers:**
   - `pyomo` - optimization modeling in Python
   - `casadi` - symbolic framework for optimal control
   - `gekko` - dynamic optimization

4. **Direct methods:**
   - Discretize `f(t)` at N time points: `f = [f₀, f₁, ..., f_N]`
   - Use gradient-based optimization with adjoint methods for gradients
   - Consider piecewise constant or piecewise linear control

**Key considerations:**
- Control variable bounds: 0 ≤ f(t) ≤ 1
- Computational cost: each objective evaluation requires forward integration
- Gradient availability: can we compute gradients via adjoint method?
- Multi-modal objective: global vs local optimization

## Project Structure

```
coin_equality/
├── README.md                          # This file
├── CLAUDE.md                          # AI coding style guide
├── requirements.txt                   # Python dependencies
├── income_distribution.py             # Core income distribution functions
├── economic_model.py                  # Forward model and ODE system
├── coin_equality (methods) v0.1.pdf   # Detailed methods document
└── [source code directories]
```

## References

Caldeira, K., Bala, G., & Cao, L. (2023). "Climate sensitivity uncertainty and the need for energy without CO₂ emission." *Nature Climate Change*.

## License

[To be specified]

## Authors

[To be specified]
