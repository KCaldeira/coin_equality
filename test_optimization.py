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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from parameters import load_configuration, ModelConfiguration
from optimization import UtilityOptimizer, create_control_function_from_points
from economic_model import integrate_model
from output import save_results, write_optimization_summary, copy_config_file


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

    if len(sys.argv) < 2:
        print("Usage: python test_optimization.py <config_file>")
        print("\nExample:")
        print("  python test_optimization.py config_baseline.json")
        sys.exit(1)

    config_path = sys.argv[1]

    config = load_configuration(config_path)
    control_times = config.optimization_params.control_times
    n_control_points = len(control_times)

    print_header(f"OPTIMIZATION TEST ({n_control_points} CONTROL POINTS)")
    print(f"Configuration file: {config_path}")
    print(f"Run name: {config.run_name}")
    print(f"Time span: {config.integration_params.t_start} to {config.integration_params.t_end} years")
    print(f"Time step: {config.integration_params.dt} years")

    optimizer = UtilityOptimizer(config)

    initial_guess = config.optimization_params.initial_guess
    max_evaluations = config.optimization_params.max_evaluations

    if len(control_times) == 1:
        print(f"\n==> Running SINGLE control point optimization")
        print(f"    Control time: {control_times[0]}")
        print(f"    Initial guess: f = {initial_guess[0]}")
        sensitivity_results = run_sensitivity_analysis(optimizer, n_points=21)
    else:
        print(f"\n==> Running MULTI-POINT control optimization")
        print(f"    Number of control points: {len(control_times)}")
        print(f"    Control times: {control_times}")
        print(f"    Initial guess: {initial_guess}")
        sensitivity_results = None

    print_header("OPTIMIZATION")
    print(f"Max evaluations: {max_evaluations}")

    opt_params = config.optimization_params
    algorithm = opt_params.algorithm if opt_params.algorithm is not None else 'LN_BOBYQA'
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

    opt_results = optimizer.optimize_control_points(
        control_times,
        initial_guess,
        max_evaluations,
        algorithm=opt_params.algorithm,
        ftol_rel=opt_params.ftol_rel,
        ftol_abs=opt_params.ftol_abs,
        xtol_rel=opt_params.xtol_rel,
        xtol_abs=opt_params.xtol_abs
    )

    if opt_results['status'] == 'degenerate':
        print(f"\n*** DEGENERATE CASE DETECTED ***")
        print(f"No income available for redistribution or abatement (deltaL = 0)")
        print(f"Control values have no effect on outcome.")
        print(f"Returning initial guess values.\n")
    else:
        print(f"Termination: {opt_results['termination_name']} (code {opt_results['termination_code']})\n")

    print_optimization_results(opt_results, len(control_times))

    if len(control_times) == 1:
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
    if len(control_times) == 1:
        print(f"Optimal constant allocation: f₀ = {f_opt:.6f}")
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
