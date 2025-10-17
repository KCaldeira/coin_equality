"""
Output functions for COIN_equality model.

Creates CSV files and PDF plots of model results in timestamped directories.
"""

import os
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def create_output_directory(run_name):
    """
    Create timestamped output directory.

    Parameters
    ----------
    run_name : str
        Name of the model run

    Returns
    -------
    str
        Path to created output directory

    Notes
    -----
    Directory format: ./data/output/{run_name}_YYYYMMDD-HHMMSS
    """
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = os.path.join('data', 'output', f'{run_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def write_results_csv(results, output_dir, filename='results.csv'):
    """
    Write results dictionary to CSV file.

    Parameters
    ----------
    results : dict
        Results dictionary from integrate_model()
    output_dir : str
        Directory to write CSV file
    filename : str
        Name of CSV file

    Returns
    -------
    str
        Path to created CSV file

    Notes
    -----
    Each column is a variable, each row is a time point.
    First row contains variable names (header).
    """
    csv_path = os.path.join(output_dir, filename)

    # Get variable names (keys) in consistent order
    var_names = sorted(results.keys())

    # Open CSV file and write
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(var_names)

        # Get number of time points
        n_points = len(results['t'])

        # Write data rows
        for i in range(n_points):
            row = [results[var][i] for var in var_names]
            writer.writerow(row)

    return csv_path


def plot_results_pdf(results, output_dir, filename='plots.pdf'):
    """
    Create PDF with time series plots of all variables.

    Parameters
    ----------
    results : dict
        Results dictionary from integrate_model()
    output_dir : str
        Directory to write PDF file
    filename : str
        Name of PDF file

    Returns
    -------
    str
        Path to created PDF file

    Notes
    -----
    Creates multi-page PDF with 6 plots per page (2 rows x 3 columns).
    Each plot shows one variable vs time.
    """
    pdf_path = os.path.join(output_dir, filename)

    # Get time array
    t = results['t']

    # Get all variable names except 't'
    var_names = sorted([k for k in results.keys() if k != 't'])

    # Create PDF
    with PdfPages(pdf_path) as pdf:
        # Process variables in groups of 6
        plots_per_page = 6
        n_vars = len(var_names)

        for page_start in range(0, n_vars, plots_per_page):
            # Create figure for this page
            fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
            fig.suptitle('COIN_equality Model Results', fontsize=14, fontweight='bold')

            # Flatten axes array for easier iteration
            axes_flat = axes.flatten()

            # Plot up to 6 variables on this page
            page_end = min(page_start + plots_per_page, n_vars)
            for i, var_idx in enumerate(range(page_start, page_end)):
                var_name = var_names[var_idx]
                ax = axes_flat[i]

                # Plot the time series
                ax.plot(t, results[var_name], linewidth=1.5)
                ax.set_xlabel('Time (yr)', fontsize=10)
                ax.set_ylabel(var_name, fontsize=10)
                ax.set_title(var_name, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Use scientific notation for large/small numbers
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))

            # Hide unused subplots on last page
            for i in range(page_end - page_start, plots_per_page):
                axes_flat[i].set_visible(False)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path


def save_results(results, run_name):
    """
    Save model results to CSV and PDF in timestamped directory.

    Parameters
    ----------
    results : dict
        Results dictionary from integrate_model()
    run_name : str
        Name of the model run

    Returns
    -------
    dict
        Dictionary with paths:
        - 'output_dir': path to output directory
        - 'csv_file': path to CSV file
        - 'pdf_file': path to PDF file

    Notes
    -----
    Creates directory: ./data/output/{run_name}_YYYYMMDD-HHMMSS
    Writes two files:
    - results.csv: all variables in tabular format
    - plots.pdf: time series plots (6 per page)
    """
    # Create output directory
    output_dir = create_output_directory(run_name)

    # Write CSV
    csv_file = write_results_csv(results, output_dir)

    # Create plots PDF
    pdf_file = plot_results_pdf(results, output_dir)

    return {
        'output_dir': output_dir,
        'csv_file': csv_file,
        'pdf_file': pdf_file,
    }
