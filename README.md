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
3. **Ω** from ΔT, k_damage_coeff, k_damage_exp (Eq 1.2: climate damage fraction)
4. **Y_damaged** from Y_gross, Ω (Eq 1.3: production after climate damage)
5. **y** from Y_damaged, L, s (Eq 1.4: mean per-capita income)
6. **Δc** from y, ΔL (Eq 4.3: per-capita amount redistributable)
7. **E_pot** from σ, Y_gross (Eq 2.1: potential emissions)
8. **abatecost** from f, Δc, L (Eq 1.5: abatement expenditure)
9. **μ** from abatecost, θ₁, θ₂, E_pot (Eq 1.6: fraction of emissions abated)
10. **Λ** from abatecost, Y_damaged (Eq 1.7: abatement cost fraction)
11. **Y_net** from Y_damaged, Λ (Eq 1.8: production after abatement costs)
12. **y_eff** from y, abatecost, L (Eq 1.9: effective per-capita income)
13. **G_eff** from f, ΔL, G₁ (Eq 4.4: effective Gini index)
14. **U** from y_eff, G_eff, η (Eq 3.5: mean utility)
15. **E** from σ, μ, Y_gross (Eq 2.3: actual emissions after abatement)
16. **dK/dt** from s, Y_net, δ, K (Eq 1.10: capital tendency)
17. **dEcum/dt = E** (cumulative emissions tendency)

### Core Components

#### 1. Economic Model (Solow-Swann Growth)

**Eq. (1.1) - Production Function (Cobb-Douglas):**
```
Y_gross(t) = A(t) · K(t)^α · L(t)^(1-α)
```

**Eq. (1.2) - Climate Damage:**
```
Ω(t) = k_damage_coeff · ΔT(t)^k_damage_exp
```
where `Ω(t)` is the fraction of gross production lost to climate damage.

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
μ(t) = [abatecost(t) · θ₂ / (E_pot(t) · θ₁(t))]^(1/θ₂)
```
The fraction of potential emissions that are abated, where:
- `E_pot(t) = σ(t) · Y_gross(t)` = potential (unabated) emissions
- `θ₁(t)` = marginal cost of abatement as μ→1 ($ tCO₂⁻¹)
- `θ₂` = abatement cost exponent

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

## Key Parameters

Parameters are organized into groups as specified in the JSON configuration file.

### Scalar Parameters (Time-Invariant)

Economic parameters:

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| `α` | Output elasticity of capital (capital share of income) | - | `alpha` |
| `δ` | Capital depreciation rate | yr⁻¹ | `delta` |
| `s` | Savings rate (fraction of net production saved) | - | `s` |

Climate parameters:

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| `k_damage_coeff` | Climate damage coefficient | °C⁻ᵏ_ᵈᵃᵐᵃᵍᵉ_ᵉˣᵖ | `k_damage_coeff` |
| `k_damage_exp` | Climate damage exponent | - | `k_damage_exp` |
| `k_climate` | Temperature sensitivity to cumulative emissions | °C tCO₂⁻¹ | `k_climate` |

Utility and inequality parameters:

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| `η` | Coefficient of relative risk aversion (CRRA) | - | `eta` |
| `ρ` | Pure rate of time preference | yr⁻¹ | `rho` |
| `G₁` | Initial Gini index (0 = perfect equality, 1 = max inequality) | - | `Gini_initial` |
| `Gini_fract` | Fraction of effective Gini change as instantaneous step (0 = no step, 1 = full step) | - | `Gini_fract` |
| `Gini_restore` | Rate at which Gini restores to initial value (0 = no restoration) | yr⁻¹ | `Gini_restore` |
| `ΔL` | Fraction of income available for redistribution | - | `deltaL` |

Abatement cost parameters:

| Parameter | Description | Units | JSON Key |
|-----------|-------------|-------|----------|
| `θ₂` | Abatement cost exponent | - | `theta2` |

### Time-Dependent Functions

These functions are evaluated at each time step:

| Function | Description | Units | JSON Key |
|----------|-------------|-------|----------|
| `A(t)` | Total factor productivity | - | `A` |
| `L(t)` | Population | people | `L` |
| `σ(t)` | Carbon intensity of GDP | tCO₂ $⁻¹ | `sigma` |
| `θ₁(t)` | Marginal abatement cost as μ→1 | $ tCO₂⁻¹ | `theta1` |

Each function is specified by `type` and type-specific parameters (e.g., `initial_value`, `growth_rate`).

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
   - Climate: `k_damage_coeff`, `k_damage_exp`, `k_climate`
   - Utility: `eta`, `rho`
   - Distribution: `Gini_initial`, `deltaL`
   - Abatement: `theta2`

4. **`time_functions`** - Time-dependent functions (A, L, sigma, theta1), each specified with:
   - `type`: "constant", "exponential_growth", "logistic_growth", or "piecewise_linear"
   - Type-specific parameters (e.g., `initial_value`, `growth_rate`)

5. **`integration_parameters`** - Solver configuration:
   - `t_start`, `t_end`, `dt`, `rtol`, `atol`

6. **`control_function`** - Allocation policy f(t):
   - `type`: "constant" or "piecewise_constant"
   - Type-specific parameters (e.g., `value` for constant)

### Adding Comments with `_description` Fields

Any JSON key starting with `_` (underscore) is treated as a comment/description and ignored during loading. This allows you to document parameters directly in the configuration file:

```json
"scalar_parameters": {
  "_description": "Time-invariant model parameters",
  "alpha": 0.3,
  "_alpha": "Capital share of income",
  "delta": 0.10,
  "_delta": "Capital depreciation rate (10% per year)"
}
```

You can add descriptions at any level:
- **Section level**: `"_description"` to describe a whole section
- **Parameter level**: `"_parameter_name"` to describe individual parameters
- **Nested levels**: Works in nested dictionaries like time functions

See `config_baseline.json` for extensive examples of documentation.

### Initial Conditions

Initial conditions are **computed automatically** (not specified in JSON):

- **`Ecum(0) = 0`**: No cumulative emissions at start
- **`K(0)`**: Steady-state capital stock with no climate damage or abatement:
  ```
  K₀ = (s · A(0) / δ)^(1/(1-α)) · L(0)
  ```

This ensures the model starts from a consistent economic equilibrium.

### Example Configuration

See `config_baseline.json` for a complete example. To create new scenarios, copy and modify this file.

### Loading Configuration

```python
from parameters import load_configuration

config = load_configuration('config_baseline.json')
# config.run_name contains the run identifier
# config.scalar_params, config.time_functions, etc. are populated
```

The `evaluate_params_at_time(t, config)` helper combines all parameters into a dict for use with `calculate_tendencies()`.

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

#### Troubleshooting

**Common issues:**
- **Missing config file**: Ensure the JSON file exists and path is correct
- **JSON syntax errors**: Validate JSON syntax in configuration file
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **Permission errors**: Ensure write access to `./data/output/` directory

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

The JSON configuration supports both single and multi-point optimization through the `optimization_parameters` section:

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
- `max_evaluations`: Maximum objective function evaluations
  - Single-point: ~1000 typically sufficient
  - Multi-point: scale with problem size (e.g., 10000 for 5 points)

## Next Steps

The following tasks are prioritized to prepare the model for production use and publication:

### 1. Update Methods Section of Paper

Revise and update the Methods section of the paper to ensure it accurately reflects the current implementation as documented in this README and the model code. The paper should provide a clear, consistent description of all model equations, parameter definitions, and computational approaches used in the codebase.

### 2. Comprehensive Code Validation

Perform a detailed verification of model calculations by manually tracing through one complete time step of the integration:
- Use the output `results.csv` file to verify intermediate calculations
- Check that all equations are implemented correctly and consistently with documentation
- Validate state variable updates, tendency calculations, and functional relationships
- Ensure numerical values propagate correctly through the calculation order
- Document any discrepancies or unexpected behaviors

This step-by-step verification will provide confidence in the correctness of the implementation.

### 3. Investigate and Improve Optimization Methods

Address current issues with the optimization routine where the model allocates resources to emissions abatement even when that produces no benefit to utility. In cases where abatement provides no utility gains, those resources should be directed toward reducing the Gini index to increase aggregate utility.

Specific investigations:
- Diagnose why the optimizer is selecting suboptimal allocation strategies
- Examine objective function gradients and sensitivity to control variables
- Consider alternative optimization algorithms or improved starting conditions
- Verify that the objective function correctly captures the trade-offs between abatement and redistribution
- Test optimizer performance across different parameter regimes

### 4. Production Code Readiness

Combine the results of code validation (Step 2) and optimization improvements (Step 3) to establish confidence that the model is ready for production use. This includes:
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
