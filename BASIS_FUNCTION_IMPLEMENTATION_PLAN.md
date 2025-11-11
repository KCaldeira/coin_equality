# **DETAILED IMPLEMENTATION PLAN: Basis Function Optimization Integration**

## **Overview**

Add a new method `optimize_with_basis_functions` to the `UtilityOptimizer` class that uses basis function representation (Chebyshev/Legendre polynomials) instead of control point interpolation. This method will support iterative refinement by progressively adding higher-order basis functions.

## **Key Requirements (UPDATED)**

1. **NO default values**: All parameters (f_min, f_max, s_min, s_max, n_iterations, n_basis_initial_f/s) must be explicitly provided - code should FAIL if missing
2. **Proper initialization**: Use T0 (constant Chebyshev) coefficient via inverse sigmoid transformation, all other coefficients = 0 on first iteration
3. **Sigmoid transformation**: Ensures f(t) and s(t) remain in bounds automatically without explicit constraints
4. **Time integration**: Each objective evaluation performs complete forward integration of economic model from t_start to t_end
5. **Spectral refinement**: Progressive addition of higher-order basis functions (frequency components), NOT spatial point refinement

---

## **Current Codebase Structure**

### **Key Classes and Methods**

1. **`UtilityOptimizer` class** (optimization.py:247-948)
   - **Attributes**:

     - `self.base_config` : ModelConfiguration object
     - `self.n_evaluations` : Counter for objective evaluations
     - `self.best_objective` : Tracks best objective found
     - `self.best_control_values` : Tracks best control values found

   - **Key method**: `calculate_objective(control_values, control_times, s_control_values, s_control_times)`

     - Takes control point arrays
     - Creates control function from control points using PCHIP interpolation
     - Integrates model with `integrate_model(config, store_detailed_output=False)`

     - Computes objective: `∫ e^(-ρt) · U(t) · L(t) dt` using trapezoidal integration
     - Returns scalar objective value

2. **`BasisControlFunction` class** (basis_control.py:18-243)
   - Represents f(t) or s(t) as: `f(t) = f_min + (f_max - f_min) * sigmoid(Σ c_i * φ_i(τ))`

   - Methods:

     - `evaluate(t, coefficients)` : Evaluate control at time t with given coefficients
     - `create_initial_guess(initial_value)` : Create initial coefficient guess for constant function

3. **Helper functions** (basis_control.py:245-308)
   - `create_dual_basis_control(t_start, t_end, n_basis_f, n_basis_s, ...)` : Creates f and s basis controls + initial coefficients
   - `basis_control_function_factory(f_control, s_control)` : Creates callable `control_func(t, coefficients)` that returns `(f, s)`

---

## **Implementation Details**

### **Method Signature**

```python
def optimize_with_basis_functions(self, n_basis_final_f, n_basis_final_s,
                                 initial_f, initial_s,
                                 max_evaluations,
                                 f_min, f_max,
                                 s_min, s_max,
                                 basis_type,
                                 n_iterations,
                                 n_basis_initial_f,
                                 n_basis_initial_s,
                                 algorithm=None,
                                 ftol_rel=None, ftol_abs=None,
                                 xtol_rel=None, xtol_abs=None):
```

**Note**: All parameters except algorithm and tolerances are REQUIRED (no defaults). The code should fail if any are not specified.

### **Algorithm Flow**

#### **STEP 1: Setup and Parameter Initialization**

1. Extract time span from `self.base_config.integration_params.t_start` and `.t_end`
2. Set default algorithm to `'LN_SBPLX'` if None
3. Verify all required parameters are provided (n_basis_initial_f, n_basis_initial_s, n_iterations, etc.)
   - NO defaults - all must be explicitly specified
   - If any required parameter is missing, raise an error
4. Calculate refinement base (parallel to control point logic):
   - If `n_iterations > 1` : `refinement_base = (n_final / n_initial)^(1/(n_iterations - 1))`
   - If `n_iterations == 1` : refinement_base not needed

#### **STEP 2: Iteration Loop**

For each iteration `k = 1, 2, ..., n_iterations` :

**A. Calculate current number of basis functions:**

```python
n_current_f = round(n_basis_initial_f * refinement_base_f^(k-1))
n_current_s = round(n_basis_initial_s * refinement_base_s^(k-1))
```

**B. Create basis control objects and initialize coefficients:**

```python
# Create basis control function objects
f_control = BasisControlFunction(
    t_start, t_end, n_current_f,
    f_min=f_min, f_max=f_max,
    basis_type=basis_type,
    use_sigmoid=True
)
s_control = BasisControlFunction(
    t_start, t_end, n_current_s,
    s_min=s_min, s_max=s_max,
    basis_type=basis_type,
    use_sigmoid=True
)

# Initialize coefficients for first iteration
if iteration == 1:
    # Use BasisControlFunction.create_initial_guess() which:
    # - Computes inverse sigmoid: g = log(target / (1 - target))
    #   where target = (initial_value - f_min) / (f_max - f_min)
    # - Sets c_0 = g (the T0 coefficient for constant function)
    # - Sets all other coefficients to 0
    # This ensures f(t) = initial_f for all t
    f_coeffs_init = f_control.create_initial_guess(initial_f)
    s_coeffs_init = s_control.create_initial_guess(initial_s)
    initial_coeffs = np.concatenate([f_coeffs_init, s_coeffs_init])
# Note: warm-starting for iteration > 1 is handled in section C
```

**Transformation details**:
- With sigmoid: `f(t) = f_min + (f_max - f_min) * sigmoid(Σ c_i * φ_i(τ))`
- For constant function with f = initial_f:
  - Need: `sigmoid(c_0 * φ_0(τ)) = (initial_f - f_min) / (f_max - f_min)` for all τ
  - Since `φ_0(τ) = T_0(τ) = 1` (constant), we need: `sigmoid(c_0) = target`
  - Inverse sigmoid: `c_0 = log(target / (1 - target))`
  - All other `c_i = 0` for i > 0

**C. Warm-start from previous iteration (if k > 1):**

```python
if iteration > 1:
    # Pad previous coefficients with zeros for new basis functions
    prev_coeffs_f = prev_optimal_f_coefficients
    prev_coeffs_s = prev_optimal_s_coefficients

    padded_f = np.zeros(n_current_f)
    padded_f[:len(prev_coeffs_f)] = prev_coeffs_f

    padded_s = np.zeros(n_current_s)
    padded_s[:len(prev_coeffs_s)] = prev_coeffs_s

    initial_coeffs = np.concatenate([padded_f, padded_s])
```

**D. Define objective function for NLopt:**

**CRITICAL**: The objective function must perform a complete forward integration of the economic model over time, evolving the system from t_start to t_end using the differential equations with the given control trajectory.

```python
def objective_function(coefficients, grad):
    """
    Objective function for NLopt (minimization).

    For each set of coefficients, this function:
    1. Creates control functions f(t) and s(t) from the coefficients
    2. Integrates the economic model FORWARD IN TIME from t_start to t_end
       using the system of differential equations (dK/dt, dE_cum/dt, etc.)
    3. Computes the discounted utility integral over the entire trajectory
    4. Returns the negative (since NLopt minimizes)
    """
    # Create control function that uses these coefficients
    control_func = basis_control_function_factory(f_control, s_control)

    def time_control(t):
        return control_func(t, coefficients)

    # Create model configuration with this control
    opt_config = ModelConfiguration(
        run_name=self.base_config.run_name,
        scalar_params=self.base_config.scalar_params,
        time_functions=self.base_config.time_functions,
        integration_params=self.base_config.integration_params,
        optimization_params=self.base_config.optimization_params,
        initial_state=self.base_config.initial_state,
        control_function=time_control
    )

    # INTEGRATE THE MODEL: This performs the iterative time evolution
    # solving the differential equations at each time step from t_start to t_end
    results = integrate_model(opt_config, store_detailed_output=False)

    # Compute objective from the complete time-integrated results
    rho = self.base_config.scalar_params.rho
    t = results['t']      # Time points from integration
    U = results['U']      # Mean utility at each time point
    L = results['L']      # Population at each time point

    discount_factors = np.exp(-rho * t)
    integrand = discount_factors * U * L
    obj = np.trapezoid(integrand, t)  # Integrate over time

    # Update tracking
    self.n_evaluations += 1
    if obj > self.best_objective:
        self.best_objective = obj

    # Return NEGATIVE because NLopt minimizes
    return -obj
```

**E. Set up and run NLopt optimizer:**

```python
n_vars = n_current_f + n_current_s
opt = nlopt.opt(nlopt.LN_SBPLX, n_vars)  # or whatever algorithm specified

opt.set_max_objective(objective_function)
opt.set_maxeval(max_evaluations)

# Wide coefficient bounds since sigmoid handles physical bounds
opt.set_lower_bounds(np.full(n_vars, -20.0))
opt.set_upper_bounds(np.full(n_vars, 20.0))

# Set tolerances if provided
if ftol_rel is not None:
    opt.set_ftol_rel(ftol_rel)
if ftol_abs is not None:
    opt.set_ftol_abs(ftol_abs)
if xtol_rel is not None:
    opt.set_xtol_rel(xtol_rel)
if xtol_abs is not None:
    opt.set_xtol_abs(xtol_abs)

# Optimize
optimal_coeffs = opt.optimize(initial_coeffs)
optimal_obj = opt.last_optimum_value()  # This is already negated back
```

**E. Store iteration results:**

```python
iteration_result = {
    'optimal_coefficients': optimal_coeffs,
    'optimal_f_coefficients': optimal_coeffs[:n_current_f],
    'optimal_s_coefficients': optimal_coeffs[n_current_f:],
    'optimal_objective': optimal_obj,
    'n_basis_f': n_current_f,
    'n_basis_s': n_current_s,
    'n_evaluations': self.n_evaluations,
    'iteration': iteration
}
iteration_history.append(iteration_result)
```

#### **STEP 3: Return Results**

Return dictionary matching the structure of `optimize_with_iterative_refinement` :

```python
return {
    'optimal_coefficients': final_optimal_coeffs,  # Full concatenated array
    'optimal_f_coefficients': final_f_coeffs,      # Just f coefficients
    'optimal_s_coefficients': final_s_coeffs,      # Just s coefficients
    'optimal_objective': final_objective,
    'n_evaluations': total_evaluations,
    'status': 'success',
    'algorithm': algorithm,
    'n_iterations': n_iterations,
    'iteration_history': iteration_history,
    'f_basis_control': f_control_final,  # BasisControlFunction object
    's_basis_control': s_control_final,  # BasisControlFunction object
    'refinement_base_f': refinement_base_f,
    'refinement_base_s': refinement_base_s
}
```

---

## **Key Differences from Control Point Method**

1. **No control times**: Basis functions represent the entire trajectory, not discrete points
2. **Different warm-start strategy**: Pad with zeros instead of interpolating
3. **Sigmoid transformation for automatic bound satisfaction**:
   - Control points use explicit bounds: `opt.set_lower_bounds([f_min, ...])` and `opt.set_upper_bounds([f_max, ...])`
   - Basis functions use sigmoid: `f(t) = f_min + (f_max - f_min) * sigmoid(Σ c_i * φ_i(τ))`
   - This means f(t) is ALWAYS in [f_min, f_max] regardless of coefficient values
   - Coefficients can have wide bounds `[-20, 20]` because physical bounds are enforced by sigmoid
4. **Objective computation**: Cannot use `calculate_objective()` because it expects control_values/control_times. Must compute objective directly from integration results (see Step 2D for details on integration).
5. **Spectral refinement**: Adding basis functions adds frequency components, NOT spatial resolution
   - Control point refinement: adds more points in time (spatial resolution)
   - Basis function refinement: adds higher-order polynomials (frequency components)

---

## **Error Handling**

1. **Degenerate case check**: Check if `fract_gdp ≈ 0` (same as existing methods)
2. **NLopt errors**: Wrap optimization in try-except to catch failures
3. **Assertion checks**: Verify coefficient array lengths match expected dimensions

---

## **Files to Modify**

### **Files on main branch** (current clean state):
- optimization.py
- test_optimization.py
- parameters.py
- basis_control.py (DOES NOT EXIST on main branch - needs to be created)

### **Required changes**:

1. **basis_control.py** (CREATE NEW FILE):
   - Implement `BasisControlFunction` class with sigmoid transformation
   - Implement helper functions: `create_dual_basis_control()`, `basis_control_function_factory()`

2. **optimization.py**:
   - Add `optimize_with_basis_functions()` method to `UtilityOptimizer` class
   - Location: After `optimize_with_iterative_refinement()` method

3. **parameters.py**:
   - Add basis function parameters to `OptimizationParameters` dataclass
   - Add `uses_basis_functions()` helper method

4. **test_optimization.py**:
   - Add dispatch logic for basis function mode (around line 519)

---

## **Testing Plan**

1. Run with `config_basis_test.json`:
   - 2 iterations
   - Start with 5 basis functions → end with 15
   - Verify warm-starting works correctly
   - Check that objective improves across iterations

2. Compare with control point method:
   - Same initial guess
   - Check final objective values are similar
   - Verify basis function method uses fewer decision variables

---

## **Questions for Review**

1. ✅ Is the objective function calculation correct (computing directly from integration results)?
2. ✅ Are the coefficient bounds `[-20, 20]` appropriate for sigmoid transformation?
3. ✅ Should I track `best_control_values` for basis function mode? (Not directly applicable since we have coefficients, not control values)
4. ✅ Is the return dictionary structure acceptable? (Mirrors control point method where appropriate)

---

---

## **Repository State**

- **Current branch**: `basis-function-optimization`
- **Branch status**: Reset to match `main` branch (commit: de79a8f "debugging")
- **Only file on branch**: `BASIS_FUNCTION_IMPLEMENTATION_PLAN.md` (untracked)
- **Clean state**: All other files match main branch exactly

---

## **END OF PLAN**

Please review this updated plan and let me know if:
1. The approach is correct (especially the sigmoid transformation and initialization)
2. Any details need clarification or modification
3. I should proceed with implementation

**Key updates based on feedback**:
- ✅ Removed all default values for required parameters
- ✅ Clarified T0 coefficient initialization using inverse sigmoid
- ✅ Emphasized time integration requirement
- ✅ Explained sigmoid transformation for automatic bound satisfaction
- ✅ Repository reset to clean main branch state
