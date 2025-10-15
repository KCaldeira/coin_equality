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

### Core Components

#### 1. Economic Model (Solow-Swann Growth)

**Production Function (Cobb-Douglas):**
```
Y_gross(t) = A(t) · K(t)^α · L(t)^(1-α)
```

**Climate Damage:**
```
Y_damaged(t) = (1 - Ω(t)) · Y_gross(t)
Ω(t) = k_damage · ΔT(t)^β
```

**Abatement Cost:**
```
Y_net(t) = (1 - Λ(t)) · Y_damaged(t)
Λ(t) = θ₁(t) · μ(t)^θ₂
```

**Capital Accumulation:**
```
dK/dt = s · Y_net(t) - δ · K(t)
```

**Per-Capita Consumption:**
```
c(t) = (1 - s) · Y_net(t) / L(t)
```

#### 2. Climate Model

**Emissions:**
```
E_base(t) = σ(t) · Y_gross(t)
E(t) = σ(t) · (1 - μ(t)) · Y_gross(t)
```

**Temperature Change:**
```
ΔT(t) = k_climate · ∫₀^t E(t') dt'
```

Temperature change is proportional to cumulative carbon dioxide emissions.

#### 3. Income Distribution and Utility

**Pareto-Lorenz Distribution:**
```
ℒ(F) = 1 - (1 - F)^(1-1/a)
```

where `F` is the fraction of the population with the lowest incomes.

**Gini Index:**
```
G = 1/(2a - 1)
a = (1 + 1/G)/2
```

**Income at Rank F:**
```
c(F) = y · (1 - 1/a) · (1 - F)^(-1/a)
```

**Isoelastic Utility Function (CRRA):**
```
u(c) = (c^(1-η) - 1)/(1 - η)  for η ≠ 1
u(c) = ln(c)                   for η = 1
```

where `η` is the coefficient of relative risk aversion.

**Mean Population Utility:**
```
U = [y^(1-η)/(1-η)] · [(1+G)^η(1-G)^(1-η)/(1+G(2η-1))]^(1/(1-η))  for η ≠ 1
U = ln(y) + ln((1-G)/(1+G)) + 2G/(1+G)                              for η = 1
```

#### 4. Redistribution Mechanics

**Crossing Rank (no income change):**
```
F* = 1 - [((1+G₁)(1-G₂))/((1-G₁)(1+G₂))]^((G₂-G₁)/(2(G₂-G₁)))
```

**Fraction of Income Redistributed:**
```
ΔL(F*) = [2(G₁-G₂)/(1-G₁)(1+G₂)] · [((1+G₁)(1-G₂))/((1-G₁)(1+G₂))]^((1+G₁)(1-G₂)/(2(G₂-G₁)))
```

**Per-Capita Amount Redistributed:**
```
Δc(F*) = y · ΔL(F*)
```

**New Gini Index with Partial Allocation to Abatement:**
```
G₂(f) = (1-ΔL)/(1-f·ΔL) · [1 - (1 - G₁)^((1-ΔL(1-F*))/(1-ΔL))]
```

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

## Project Structure

```
coin_equality/
├── README.md                          # This file
├── CLAUDE.md                          # AI coding style guide
├── coin_equality (methods) v0.1.pdf   # Detailed methods document
└── [source code directories]
```

## References

Caldeira, K., Bala, G., & Cao, L. (2023). "Climate sensitivity uncertainty and the need for energy without CO₂ emission." *Nature Climate Change*.

## License

[To be specified]

## Authors

[To be specified]
