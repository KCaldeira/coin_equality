"""
Test that basis function evaluation produces the expected initial values.
"""

import numpy as np
from basis_control import BasisControlFunction

# Test parameters matching config
t_start, t_end = 0.0, 400.0
n_basis = 5
basis_type = 'chebyshev'
eps = 1e-10

# Test f (abatement fraction)
f_min, f_max = 0.0, 1.0
initial_f = 0.5

f_control = BasisControlFunction(
    t_start, t_end, n_basis,
    value_min=f_min, value_max=f_max,
    basis_type=basis_type, eps=eps
)

f_coeffs = f_control.create_initial_guess(initial_f)
print(f"f coefficients: {f_coeffs}")
print(f"  c_0 = {f_coeffs[0]:.6f}")

# Evaluate at several time points
test_times = [0.0, 100.0, 200.0, 300.0, 400.0]
print(f"\nEvaluating f(t) at various times:")
for t in test_times:
    f_val = f_control.evaluate(t, f_coeffs)
    error = abs(f_val - initial_f)
    print(f"  t={t:6.1f}: f(t)={f_val:.6f} (expected {initial_f}, error={error:.2e})")

# Test s (savings rate)
s_min, s_max = 0.0, 1.0
initial_s = 0.28

s_control = BasisControlFunction(
    t_start, t_end, n_basis,
    value_min=s_min, value_max=s_max,
    basis_type=basis_type, eps=eps
)

s_coeffs = s_control.create_initial_guess(initial_s)
print(f"\ns coefficients: {s_coeffs}")
print(f"  c_0 = {s_coeffs[0]:.6f}")

print(f"\nEvaluating s(t) at various times:")
for t in test_times:
    s_val = s_control.evaluate(t, s_coeffs)
    error = abs(s_val - initial_s)
    print(f"  t={t:6.1f}: s(t)={s_val:.6f} (expected {initial_s}, error={error:.2e})")

# Test combined evaluation (as done in optimization)
combined_coeffs = np.concatenate([f_coeffs, s_coeffs])
print(f"\nCombined coefficients ({len(combined_coeffs)} total):")
print(f"  f coeffs: {f_coeffs}")
print(f"  s coeffs: {s_coeffs}")

print(f"\nCombined evaluation:")
for t in test_times:
    f_val = f_control.evaluate(t, combined_coeffs[:n_basis])
    s_val = s_control.evaluate(t, combined_coeffs[n_basis:])
    print(f"  t={t:6.1f}: f={f_val:.6f}, s={s_val:.6f}")
