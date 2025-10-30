#!/usr/bin/env python
"""Simple test to isolate the issue."""

print("Starting...")

from parameters import load_configuration
print("1. Imports successful")

config = load_configuration('config_test_DICE_2x_0.02_10k_0.67.json')
print("2. Config loaded")

print(f"   n_points_final_f: {config.optimization_params.n_points_final_f}")
print(f"   n_points_final_s: {config.optimization_params.n_points_final_s}")
print(f"   initial_guess_s: {config.optimization_params.initial_guess_s}")

from optimization import UtilityOptimizer
print("3. Optimizer imported")

optimizer = UtilityOptimizer(config)
print("4. Optimizer created")

print("\nTest completed successfully!")
