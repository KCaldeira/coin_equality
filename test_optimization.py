"""
Test script for optimization with single or multiple control points.

This script demonstrates the complete workflow:
1. Load baseline configuration
2. Perform sensitivity analysis (for single control point only)
3. Run optimization (handles both single and multiple control points)
4. Compare optimal vs. arbitrary constant allocations (for single control point)
5. Generate visualization plots
"""

import sys
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from parameters import load_configuration, ModelConfiguration
from optimization import UtilityOptimizer, create_control_function_from_points
from economic_model import integrate_model
from output import save_results, write_optimization_summary, copy_config_file


def apply_config_override(config_dict, key_path, value):
    """
    Apply a command line override to a nested configuration dictionary.

    Parameters
    ----------
    config_dict : dict
        The configuration dictionary to modify
    key_path : str
        Dot-separated path to the key (e.g., "scalar_parameters.alpha")
    value : str
        String value to set (will be converted to appropriate type)
    """
    keys = key_path.split('.')

    # Navigate to the parent dict
    current = config_dict
    for key in keys[:-1]:
        if key not in current:
            raise KeyError(f"Key path '{key_path}' not found in config (failed at '{key}')")
        current = current[key]

    final_key = keys[-1]
    if final_key not in current:
        raise KeyError(f"Key path '{key_path}' not found in config (final key '{final_key}' not found)")

    # Infer type from existing value
    existing_value = current[final_key]

    try:
        if existing_value is None:
            # If existing is None, try int -> float -> bool -> string
            try:
                converted_value = int(value)
            except ValueError:
                try:
                    converted_value = float(value)
                except ValueError:
                    if value.lower() in ('true', 'false'):
                        converted_value = value.lower() == 'true'
                    else:
                        converted_value = value
        elif isinstance(existing_value, bool):
            # Handle bools before int (since bool is subclass of int in Python)
            converted_value = value.lower() in ('true', '1', 'yes')
        elif isinstance(existing_value, int):
            converted_value = int(value)
        elif isinstance(existing_value, float):
            converted_value = float(value)
        elif isinstance(existing_value, str):
            converted_value = value
        elif isinstance(existing_value, (list, dict)):
            # Try to parse as JSON for lists and dicts
            converted_value = json.loads(value)
        else:
            # Fallback: keep as string
            converted_value = value
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"Cannot convert '{value}' to type {type(existing_value).__name__} for key '{key_path}': {e}")

    current[final_key] = converted_value
    print(f"Override: {key_path} = {converted_value} (was {existing_value})")


def parse_arguments():
    """
    Parse command line arguments including config file and overrides.

    Returns
    -------
    tuple
        (config_path, overrides_dict) where overrides_dict maps key paths to values
    """
    parser = argparse.ArgumentParser(
        description='Run optimization with optional parameter overrides',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_optimization.py config_baseline.json
  python test_optimization.py config_baseline.json --scalar_parameters.alpha 0.35
  python test_optimization.py config_baseline.json --optimization_parameters.initial_guess_f 0.3 --run_name "test_run"

Override format:
  --key.subkey.subsubkey value

Common overrides:
  --run_name <name>
  --scalar_parameters.alpha <value>
  --scalar_parameters.rho <value>
  --optimization_parameters.initial_guess_f <value>
  --optimization_parameters.initial_guess_s <value>
  --optimization_parameters.max_evaluations <value>
  --optimization_parameters.n_points_final_f <value>
  --optimization_parameters.n_points_final_s <value>
  --time_functions.A.growth_rate <value>
        """
    )

    parser.add_argument('config_file', help='Path to JSON configuration file')

    # Use parse_known_args to allow arbitrary --key value pairs
    args, unknown = parser.parse_known_args()

    # Parse overrides from unknown arguments
    overrides = {}
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith('--'):
            key = arg[2:]  # Remove '--' prefix
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                value = unknown[i + 1]
                overrides[key] = value
                i += 2
            else:
                raise ValueError(f"Override '{arg}' requires a value")
        else:
            raise ValueError(f"Unexpected argument '{arg}'. Overrides must start with '--'")

    return args.config_file, overrides


def print_header(text):
    """Print formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}\n")


def print_optimization_results(opt_results, n_control_points):
    """Print formatted optimization results."""
    if n_control_points == 1:
        print(f"Optimal f₀:             {opt_results['optimal_values'][0]:.6f}")
    else:
        print(f"Optimal control values:")
        for t, f_val in opt_results['control_points']:
            print(f"  t={t:6.1f} yr: f={f_val:.6f}")
    print(f"Optimal objective:      {opt_results['optimal_objective']:.6e}")
    print(f"Function evaluations:   {opt_results['n_evaluations']}")
    print(f"Status:                 {opt_results['status']}")


def run_sensitivity_analysis(optimizer, n_points=21):
    """
    Run sensitivity analysis across f values from 0 to 1.

    Parameters
    ----------
    optimizer : UtilityOptimizer
        Optimizer instance
    n_points : int
        Number of points to evaluate (default: 21)

    Returns
    -------
    dict
        Sensitivity analysis results
    """
    print_header("SENSITIVITY ANALYSIS")
    print(f"Evaluating objective function at {n_points} points from f=0 to f=1...")

    f_values = np.linspace(0, 1, n_points)
    results = optimizer.sensitivity_analysis(f_values)

    print(f"\nCompleted {results['n_evaluations']} evaluations")
    print(f"\nObjective value range:")
    print(f"  Minimum: {results['objectives'].min():.6e} at f={f_values[results['objectives'].argmin()]:.3f}")
    print(f"  Maximum: {results['objectives'].max():.6e} at f={f_values[results['objectives'].argmax()]:.3f}")

    return results




def compare_scenarios(config, f_values, labels):
    """
    Run model with different constant f values for comparison.

    Parameters
    ----------
    config : ModelConfiguration
        Base configuration
    f_values : list of float
        List of f values to compare
    labels : list of str
        Labels for each scenario

    Returns
    -------
    dict
        Dictionary mapping labels to results
    """
    print_header("COMPARING SCENARIOS")

    comparison_results = {}

    for f_val, label in zip(f_values, labels):
        print(f"Running scenario: {label} (f={f_val:.4f})...")

        scenario_config = ModelConfiguration(
            run_name=config.run_name,
            scalar_params=config.scalar_params,
            time_functions=config.time_functions,
            integration_params=config.integration_params,
            optimization_params=config.optimization_params,
            initial_state=config.initial_state,
            control_function=create_control_function_from_points([(0, f_val)])
        )

        results = integrate_model(scenario_config)

        rho = config.scalar_params.rho
        t = results['t']
        U = results['U']
        L = results['L']
        discount_factors = np.exp(-rho * t)
        integrand = discount_factors * U * L
        objective = np.trapezoid(integrand, t)

        comparison_results[label] = {
            'f': f_val,
            'objective': objective,
            'results': results
        }

        print(f"  Objective: {objective:.6e}")

    return comparison_results


def create_visualization_plots(sensitivity_results, opt_results, comparison_results, run_name, output_pdf):
    """
    Create visualization plots and save to PDF.

    Parameters
    ----------
    sensitivity_results : dict
        Results from sensitivity analysis
    opt_results : dict
        Results from optimization
    comparison_results : dict
        Results from scenario comparison
    run_name : str
        Name of the model run to display in header
    output_pdf : str
        Path to output PDF file
    """
    print_header("GENERATING VISUALIZATION PLOTS")
    print(f"Creating plots in: {output_pdf}")

    with PdfPages(output_pdf) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{run_name} - Single Control Point Optimization Results', fontsize=14, fontweight='bold')

        f_vals = sensitivity_results['f_values']
        obj_vals = sensitivity_results['objectives']
        f_opt = opt_results['optimal_values'][0]
        obj_opt = opt_results['optimal_objective']

        ax = axes[0, 0]
        ax.plot(f_vals, obj_vals, 'b-', linewidth=2, label='Objective function')
        ax.plot(f_opt, obj_opt, 'r*', markersize=15, label=f'Optimum (f={f_opt:.4f})')
        ax.set_xlabel('Abatement fraction f', fontsize=11)
        ax.set_ylabel('Discounted utility', fontsize=11)
        ax.set_title('Objective Function Landscape', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        ax = axes[0, 1]
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for (label, data), color in zip(comparison_results.items(), colors):
            t = data['results']['t']
            G_eff = data['results']['G_eff']
            ax.plot(t, G_eff, label=label, linewidth=2, color=color)
        ax.set_xlabel('Time (years)', fontsize=11)
        ax.set_ylabel('Effective Gini index', fontsize=11)
        ax.set_title('Inequality Over Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        ax = axes[1, 0]
        for (label, data), color in zip(comparison_results.items(), colors):
            t = data['results']['t']
            delta_T = data['results']['delta_T']
            ax.plot(t, delta_T, label=label, linewidth=2, color=color)
        ax.set_xlabel('Time (years)', fontsize=11)
        ax.set_ylabel('Temperature change (°C)', fontsize=11)
        ax.set_title('Climate Warming Over Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        ax = axes[1, 1]
        for (label, data), color in zip(comparison_results.items(), colors):
            t = data['results']['t']
            U = data['results']['U']
            ax.plot(t, U, label=label, linewidth=2, color=color)
        ax.set_xlabel('Time (years)', fontsize=11)
        ax.set_ylabel('Mean utility', fontsize=11)
        ax.set_title('Mean Utility Over Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{run_name} - Economic and Emissions Comparison', fontsize=14, fontweight='bold')

        ax = axes[0, 0]
        for (label, data), color in zip(comparison_results.items(), colors):
            t = data['results']['t']
            y_eff = data['results']['y_eff']
            ax.plot(t, y_eff, label=label, linewidth=2, color=color)
        ax.set_xlabel('Time (years)', fontsize=11)
        ax.set_ylabel('Effective per-capita income ($)', fontsize=11)
        ax.set_title('Per-Capita Income Over Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

        ax = axes[0, 1]
        for (label, data), color in zip(comparison_results.items(), colors):
            t = data['results']['t']
            mu = data['results']['mu']
            ax.plot(t, mu, label=label, linewidth=2, color=color)
        ax.set_xlabel('Time (years)', fontsize=11)
        ax.set_ylabel('Abatement fraction μ', fontsize=11)
        ax.set_title('Emissions Abatement Fraction', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        ax = axes[1, 0]
        for (label, data), color in zip(comparison_results.items(), colors):
            t = data['results']['t']
            E = data['results']['E']
            ax.plot(t, E, label=label, linewidth=2, color=color)
        ax.set_xlabel('Time (years)', fontsize=11)
        ax.set_ylabel('Emissions (tCO₂/yr)', fontsize=11)
        ax.set_title('CO₂ Emissions Rate Over Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

        ax = axes[1, 1]
        for (label, data), color in zip(comparison_results.items(), colors):
            t = data['results']['t']
            Lambda = data['results']['Lambda']
            ax.plot(t, Lambda, label=label, linewidth=2, color=color)
        ax.set_xlabel('Time (years)', fontsize=11)
        ax.set_ylabel('Abatement cost fraction Λ', fontsize=11)
        ax.set_title('Abatement Cost as Fraction of GDP', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"Saved visualization plots to: {output_pdf}")


def main():
    """Main execution function."""
    start_time = time.time()

    # Parse command line arguments
    config_path, overrides = parse_arguments()

    # Load base configuration from JSON file
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Apply command line overrides
    if overrides:
        print_header("APPLYING COMMAND LINE OVERRIDES")
        for key_path, value in overrides.items():
            apply_config_override(config_dict, key_path, value)

    # Create configuration object from modified dict
    # Save modified dict to temp file for load_configuration to process
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(config_dict, tmp, indent=2)
        tmp_path = tmp.name

    try:
        config = load_configuration(tmp_path)
    finally:
        os.unlink(tmp_path)
    opt_params = config.optimization_params

    is_iterative = opt_params.is_iterative_refinement()

    if is_iterative:
        n_iterations = opt_params.control_times
        print_header(f"ITERATIVE REFINEMENT OPTIMIZATION ({n_iterations} ITERATIONS)")
        print(f"Configuration file: {config_path}")
        print(f"Run name: {config.run_name}")
        print(f"Time span: {config.integration_params.t_start} to {config.integration_params.t_end} years")
        print(f"Time step: {config.integration_params.dt} years")
        print(f"\nIterative refinement mode:")
        print(f"  Number of iterations: {n_iterations}")
        if opt_params.n_points_final_f is not None:
            print(f"  Final f control points: {opt_params.n_points_final_f}")
        else:
            print(f"  Final f control points: {round(1 + 2.0**(n_iterations - 1))}")
        print(f"  Initial guess: f = {opt_params.initial_guess_f}")
        if opt_params.initial_guess_s is not None:
            print(f"  Initial guess: s = {opt_params.initial_guess_s}")
            if opt_params.n_points_final_s is not None:
                print(f"  Final s control points: {opt_params.n_points_final_s}")
            else:
                print(f"  Final s control points: same as f")
            print(f"  Optimizing both f and s: YES")
    else:
        control_times = opt_params.control_times
        n_control_points = len(control_times)
        print_header(f"DIRECT OPTIMIZATION ({n_control_points} CONTROL POINTS)")
        print(f"Configuration file: {config_path}")
        print(f"Run name: {config.run_name}")
        print(f"Time span: {config.integration_params.t_start} to {config.integration_params.t_end} years")
        print(f"Time step: {config.integration_params.dt} years")

    optimizer = UtilityOptimizer(config)

    initial_guess = opt_params.initial_guess_f
    max_evaluations = opt_params.max_evaluations

    if is_iterative:
        sensitivity_results = None
    elif len(opt_params.control_times) == 1:
        print(f"\n==> Running SINGLE control point optimization")
        print(f"    Control time: {opt_params.control_times[0]}")
        print(f"    Initial guess: f = {initial_guess[0]}")
        sensitivity_results = run_sensitivity_analysis(optimizer, n_points=21)
    else:
        print(f"\n==> Running MULTI-POINT control optimization")
        print(f"    Number of control points: {len(opt_params.control_times)}")
        print(f"    Control times: {opt_params.control_times}")
        print(f"    Initial guess: {initial_guess}")
        sensitivity_results = None

    print_header("OPTIMIZATION")
    print(f"Max evaluations: {max_evaluations}{' per iteration' if is_iterative else ''}")

    algorithm = opt_params.algorithm if opt_params.algorithm is not None else 'LN_SBPLX'
    print(f"Algorithm: {algorithm}")

    if opt_params.ftol_rel is not None:
        print(f"ftol_rel: {opt_params.ftol_rel}")
    if opt_params.ftol_abs is not None:
        print(f"ftol_abs: {opt_params.ftol_abs}")
    if opt_params.xtol_rel is not None:
        print(f"xtol_rel: {opt_params.xtol_rel}")
    if opt_params.xtol_abs is not None:
        print(f"xtol_abs: {opt_params.xtol_abs}")

    print(f"\nRunning {algorithm} optimization...\n")

    if is_iterative:
        opt_results = optimizer.optimize_with_iterative_refinement(
            n_iterations=opt_params.control_times,
            initial_guess_scalar=opt_params.initial_guess_f,
            max_evaluations=max_evaluations,
            algorithm=opt_params.algorithm,
            ftol_rel=opt_params.ftol_rel,
            ftol_abs=opt_params.ftol_abs,
            xtol_rel=opt_params.xtol_rel,
            xtol_abs=opt_params.xtol_abs,
            n_points_final=opt_params.n_points_final_f,
            initial_guess_s_scalar=opt_params.initial_guess_s,
            n_points_final_s=opt_params.n_points_final_s
        )
        n_final_control_points = len(opt_results['control_points'])
    else:
        opt_results = optimizer.optimize_control_points(
            opt_params.control_times,
            initial_guess,
            max_evaluations,
            algorithm=opt_params.algorithm,
            ftol_rel=opt_params.ftol_rel,
            ftol_abs=opt_params.ftol_abs,
            xtol_rel=opt_params.xtol_rel,
            xtol_abs=opt_params.xtol_abs
        )
        n_final_control_points = len(opt_params.control_times)

    if opt_results.get('status') == 'degenerate':
        print(f"\n*** DEGENERATE CASE DETECTED ***")
        print(f"No income available for redistribution or abatement (fract_gdp = 0)")
        print(f"Control values have no effect on outcome.")
        print(f"Returning initial guess values.\n")
    elif not is_iterative and 'termination_name' in opt_results:
        print(f"Termination: {opt_results['termination_name']} (code {opt_results['termination_code']})\n")

    print_optimization_results(opt_results, n_final_control_points)

    if not is_iterative and len(opt_params.control_times) == 1:
        f_opt = opt_results['optimal_values'][0]
        comparison_scenarios = {
            'Optimal': f_opt,
            'All redistribution (f=0)': 0.0,
            'Balanced (f=0.5)': 0.5,
            'All abatement (f=1)': 1.0,
        }
        comparison_results = compare_scenarios(
            config,
            list(comparison_scenarios.values()),
            list(comparison_scenarios.keys())
        )
    else:
        comparison_results = None

    print_header("SAVING RESULTS")

    optimal_config = ModelConfiguration(
        run_name=f"{config.run_name}_optimization",
        scalar_params=config.scalar_params,
        time_functions=config.time_functions,
        integration_params=config.integration_params,
        optimization_params=config.optimization_params,
        initial_state=config.initial_state,
        control_function=create_control_function_from_points(opt_results['control_points'])
    )

    print("Running forward model with optimal control...")
    optimal_results = integrate_model(optimal_config)

    print("\nSaving integration results (CSV and PDF)...")
    plot_short_horizon = config.integration_params.plot_short_horizon
    output_paths = save_results(optimal_results, optimal_config.run_name, plot_short_horizon)
    print(f"  Output directory: {output_paths['output_dir']}")
    print(f"  Results CSV:      {output_paths['csv_file']}")
    print(f"  Results PDF:      {output_paths['pdf_file']}")
    if 'pdf_file_short' in output_paths:
        print(f"  Short-term PDF:   {output_paths['pdf_file_short']}")

    print("\nWriting optimization summary CSV...")
    opt_csv_path = write_optimization_summary(opt_results, sensitivity_results, output_paths['output_dir'], 'optimization_summary.csv')
    print(f"  Optimization CSV: {opt_csv_path}")

    print("\nCopying configuration file...")
    config_copy_path = copy_config_file(config_path, output_paths['output_dir'])
    print(f"  Configuration:    {config_copy_path}")

    if comparison_results:
        print("\nCreating comparison visualization plots...")
        output_pdf = f'optimization_comparison_{config.run_name}.pdf'
        create_visualization_plots(sensitivity_results, opt_results, comparison_results, config.run_name, output_pdf)
        print(f"  Comparison PDF:   {output_pdf}")

    print_header("SUMMARY")
    if is_iterative:
        print(f"Iterative refinement optimization complete:")
        print(f"  Iterations performed: {opt_results['n_iterations']}")
        print(f"  Total evaluations: {opt_results['n_evaluations']}")
        print(f"  Final control points: {n_final_control_points}")
        print(f"\nOptimal objective value: {opt_results['optimal_objective']:.6e}")
        print(f"\nIteration history:")
        for iter_result in opt_results['iteration_history']:
            print(f"  Iteration {iter_result['iteration']:2d}: {iter_result['n_control_points']:3d} points, "
                  f"objective = {iter_result['optimal_objective']:.6e}, "
                  f"evals = {iter_result['n_evaluations']}")
        print(f"\nFinal control trajectory:")
        for t, f_val in opt_results['control_points']:
            print(f"  t={t:6.1f} yr: f={f_val:.6f}")
    elif not is_iterative and len(opt_params.control_times) == 1:
        print(f"Optimal constant allocation: f₀ = {opt_results['optimal_values'][0]:.6f}")
        print(f"Optimal objective value: {opt_results['optimal_objective']:.6e}")
        print(f"\nComparison with other strategies:")
        for label, data in comparison_results.items():
            obj_diff = data['objective'] - opt_results['optimal_objective']
            pct_diff = 100 * obj_diff / abs(opt_results['optimal_objective'])
            if label == 'Optimal':
                print(f"  {label:30s}: objective = {data['objective']:.6e} (optimal)")
            else:
                print(f"  {label:30s}: objective = {data['objective']:.6e} ({pct_diff:+.2f}% from optimal)")
    else:
        print(f"Optimal control trajectory:")
        for t, f_val in opt_results['control_points']:
            print(f"  t={t:6.1f} yr: f={f_val:.6f}")
        print(f"\nOptimal objective value: {opt_results['optimal_objective']:.6e}")

    print(f"\nAll results saved to: {output_paths['output_dir']}")

    elapsed_time = time.time() - start_time
    print(f"\nTotal runtime: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    print_header("OPTIMIZATION TEST COMPLETE")


if __name__ == '__main__':
    main()
