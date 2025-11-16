# Multi-Run Comparison Tool Implementation Plan

## Overview

Create a command-line tool to compare optimization results across multiple runs. The tool will accept directory paths (with optional wildcards), extract data from `optimization_summary.csv` and `results.csv` files, and generate comparative visualizations in both XLSX and PDF formats.

## Objectives

1. Create command-line interface accepting 1-3 directory path arguments with wildcard support
2. Extract and compare data from `optimization_summary.csv` files across runs
3. Extract and compare data from `results.csv` files across runs
4. Generate XLSX workbook with multi-sheet comparison tables
5. Generate PDF report with comparative scatter plots and time series
6. Refactor visualization code into reusable utility module

## Command-Line Interface

### Usage

```bash
python compare_results.py <path1> [path2] [path3] [...]
```

### Arguments

- **path1, path2, ...**: Unlimited directory paths with optional wildcards
  - Single directory: `results/run_001/`
  - Wildcard pattern: `results/run_*/` or `results/test_*_000/`
  - No limit on number of path arguments

### Examples

```bash
# Compare three specific directories
python compare_results.py results/baseline/ results/sensitivity_high/ results/sensitivity_low/

# Compare all matching directories from wildcard
python compare_results.py "results/test_*_000/"

# Mix of specific and wildcard
python compare_results.py results/baseline/ "results/sensitivity_*/"
```

### Output Files

Generated in current working directory:
- `comparison_summary.xlsx` - Multi-sheet workbook with tabular comparisons
- `comparison_plots.pdf` - PDF report with all comparative visualizations

## File Structure

### New Files

```
coin_equality/
├── compare_results.py          # Main comparison tool CLI
├── visualization_utils.py      # Refactored visualization functions
└── comparison_utils.py         # Comparison-specific utilities
```

### Modified Files

```
test_optimization.py            # Refactor to use visualization_utils
```

## Design Decisions

### 0. Unified Plotting Function Architecture

**Key Insight:** Create a single plotting function that works for both single-run and multi-run cases.

**Approach:**
- Each variable gets exactly one panel
- Single-run case: one line per panel
- Multi-run case: multiple lines per panel (one per case)
- Same function signature, same formatting, automatic consistency

**Changes to current implementation:**
- **Split dual-line panels** in `create_visualization_plots()` into separate single-line panels
- **Reserve multi-line panels** exclusively for showing different cases in comparisons
- **Unified function**: `plot_timeseries(case_data, variable, ylabel, title)` where `case_data` can have one or many cases

**Example:**

Current (dual-line panel):
```
Panel: "Temperature and Emissions"
  - Line 1: T_atm (blue, left y-axis)
  - Line 2: E (red, right y-axis)
```

New (separate panels):
```
Panel 1: "Atmospheric Temperature"
  - Line(s): T_atm for each case

Panel 2: "CO₂ Emissions"
  - Line(s): E for each case
```

**Benefits:**
- Any formatting improvements automatically apply to both single and multi-run reports
- Cleaner separation of concerns
- Multi-line panels have consistent meaning (different cases, not different variables)
- Easier to maintain and extend

### 1. Directory Discovery

**Approach:** Use `glob` module for wildcard expansion

```python
import glob
from pathlib import Path

def discover_directories(path_args):
    """
    Expand wildcard paths and return list of valid directories.

    Parameters
    ----------
    path_args : list of str
        Command-line path arguments (may contain wildcards)

    Returns
    -------
    list of Path
        Sorted list of directories containing required CSV files
    """
    directories = []
    for arg in path_args:
        matches = glob.glob(arg)
        for match in matches:
            path = Path(match)
            if path.is_dir():
                # Verify required files exist
                if (path / 'optimization_summary.csv').exists():
                    directories.append(path)

    return sorted(directories)
```

**Validation:**
- Check that each directory contains `optimization_summary.csv`
- Warn if `results.csv` is missing (skip results comparison for that case)
- Error if no valid directories found

### 2. Case Naming

Each discovered directory needs a readable name for labels/legends.

**Strategy:**
- Use directory name by default: `Path(dir).name`
- For long paths, use last 2 components: `Path(dir).parent.name / Path(dir).name`
- User can optionally provide custom labels via config file (future enhancement)

### 3. Data Extraction

#### optimization_summary.csv Structure

Expected columns (from test_optimization.py):
- Column A: Iteration
- Column B: N evaluations
- Column C: Objective
- Column D: Elapsed time (seconds)
- Column E: Termination status
- Additional columns: optimal f/s values

```python
import pandas as pd

def load_optimization_summary(directory):
    """
    Load optimization_summary.csv from directory.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: iteration, n_evaluations, objective,
        elapsed_time, termination_status, ...
    """
    csv_path = Path(directory) / 'optimization_summary.csv'
    df = pd.read_csv(csv_path)
    return df
```

#### results.csv Structure

Expected columns:
- t: Time (years)
- y: Per-capita output
- K: Capital stock
- T_atm: Atmospheric temperature
- E_cum: Cumulative emissions
- ... (many more economic and climate variables)

### 4. XLSX Output Structure

**Workbook:** `comparison_summary.xlsx`

**Sheet 1: "Objective"**
```
          Case1        Case2        Case3
Iter 1    1.53e13      1.54e13      1.52e13
Iter 2    1.55e13      1.56e13      1.54e13
...
```

**Sheet 2: "Evaluations"**
```
          Case1        Case2        Case3
Iter 1    1000         1200         950
Iter 2    3000         3200         2900
...
```

**Sheet 3: "Elapsed Time"**
```
          Case1        Case2        Case3
Iter 1    45.2         52.1         43.8
Iter 2    120.5        135.2        118.3
...
```

**Sheet 4: "Termination Status"**
```
          Case1        Case2        Case3
Iter 1    SUCCESS      SUCCESS      SUCCESS
Iter 2    SUCCESS      XTOL_REACHED SUCCESS
...
```

**Implementation:**
```python
import openpyxl
from openpyxl.styles import Font, Alignment

def create_comparison_xlsx(case_data, output_path):
    """
    Create Excel workbook with multi-sheet comparison.

    Parameters
    ----------
    case_data : dict
        {case_name: optimization_summary_df, ...}
    output_path : Path
        Output file path
    """
    wb = openpyxl.Workbook()

    # Remove default sheet
    wb.remove(wb.active)

    # Define sheets and their data columns
    sheets = [
        ('Objective', 'objective'),
        ('Evaluations', 'n_evaluations'),
        ('Elapsed Time', 'elapsed_time'),
        ('Termination Status', 'termination_status')
    ]

    for sheet_name, column_name in sheets:
        ws = wb.create_sheet(sheet_name)

        # Header row
        ws['A1'] = 'Iteration'
        for col_idx, case_name in enumerate(case_data.keys(), start=2):
            ws.cell(1, col_idx, case_name).font = Font(bold=True)

        # Data rows
        max_iterations = max(len(df) for df in case_data.values())
        for iter_num in range(1, max_iterations + 1):
            ws.cell(iter_num + 1, 1, iter_num)

            for col_idx, (case_name, df) in enumerate(case_data.items(), start=2):
                if iter_num <= len(df):
                    value = df.iloc[iter_num - 1][column_name]
                    ws.cell(iter_num + 1, col_idx, value)

    wb.save(output_path)
```

### 5. PDF Visualization Structure

**PDF Report:** `comparison_plots.pdf`

#### Page 1: Objective Comparison (Scatter Plot)

```
X-axis: Case index (1, 2, 3, ...)
Y-axis: Objective value
Symbols:
  - Circle (o): Iteration 1
  - Square (s): Iteration 2
  - Triangle (^): Iteration 3
  - Diamond (D): Iteration 4+
Colors: One per case
```

```python
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def create_objective_scatter(case_data):
    """
    Create scatter plot comparing objectives across cases and iterations.

    Parameters
    ----------
    case_data : dict
        {case_name: optimization_summary_df, ...}

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    case_names = list(case_data.keys())

    for case_idx, (case_name, df) in enumerate(case_data.items()):
        for iter_idx, row in df.iterrows():
            marker = markers[min(iter_idx, len(markers) - 1)]
            ax.scatter(
                case_idx + 1,
                row['objective'],
                marker=marker,
                s=100,
                label=f"{case_name} - Iter {iter_idx + 1}" if case_idx == 0 else None,
                alpha=0.7
            )

    ax.set_xlabel('Case')
    ax.set_ylabel('Objective Value')
    ax.set_title('Objective Comparison Across Cases')
    ax.set_xticks(range(1, len(case_names) + 1))
    ax.set_xticklabels(case_names, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

#### Page 2: Elapsed Time Comparison

Similar scatter plot structure for elapsed time.

#### Page 3+: Results Time Series Comparisons

For each variable in `results.csv`, create time series plots using the unified plotting function:
- X-axis: Time (years)
- Y-axis: Variable value
- Lines: One per case (multiple lines for multi-run, single line for single-run)
- Legend: Case names
- **Same function for both single-run and multi-run cases**

**All variables from results.csv** (comprehensive list based on current `create_visualization_plots()` and economic model outputs):

**Economic Variables:**
1. `y` - Per-capita output
2. `y_eff` - Effective per-capita income
3. `K` - Capital stock
4. `C` - Consumption
5. `I` - Investment
6. `s` or `s_interp` - Savings rate

**Climate Variables:**
7. `T_atm` - Atmospheric temperature
8. `T_ocean` - Ocean temperature
9. `delta_T` - Temperature change from pre-industrial
10. `E` - CO₂ emissions rate
11. `E_cum` - Cumulative emissions
12. `M_atm` - Atmospheric CO₂ concentration
13. `M_upper` - Upper ocean carbon
14. `M_lower` - Lower ocean carbon

**Abatement and Damage:**
15. `mu` or `f_interp` - Emissions abatement fraction
16. `Lambda` - Abatement cost fraction
17. `Omega` - Climate damage fraction
18. `Omega_1` - Damage to rich group
19. `Omega_2` - Damage to poor group

**Inequality and Utility:**
20. `G` - Gini coefficient
21. `G_eff` - Effective Gini index
22. `U` - Mean utility
23. `U_1` - Utility of rich group
24. `U_2` - Utility of poor group

**Other Control Variables:**
25. `f` - Abatement allocation fraction (if optimized)

**PDF Layout:**
- **16:9 aspect ratio** optimized for screen viewing (e.g., 1920×1080)
- **Landscape orientation**
- **6 plots per page** in a 2×3 grid
- Consistent formatting across all plots
- ~4-5 pages total for all 25 variables plus summary pages

```python
def plot_timeseries(ax, case_data, variable, ylabel, title):
    """
    Create time series plot on given axes (unified function for single/multi-run).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    case_data : dict
        {case_name: results_df, ...}
        Single case: {'Case': df}
        Multiple cases: {'Case1': df1, 'Case2': df2, ...}
    variable : str
        Column name in results.csv
    ylabel : str
        Y-axis label
    title : str
        Plot title
    """
    colors = plt.cm.tab10(range(10))

    for idx, (case_name, df) in enumerate(case_data.items()):
        if variable in df.columns:
            ax.plot(
                df['t'],
                df[variable],
                label=case_name if len(case_data) > 1 else None,
                linewidth=2,
                color=colors[idx % 10]
            )

    ax.set_xlabel('Time (years)', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    if len(case_data) > 1:
        ax.legend(loc='best', fontsize=8)


def create_results_report_pdf(case_data, output_pdf):
    """
    Create PDF report with all time series plots (6 per page, landscape).

    Parameters
    ----------
    case_data : dict
        {case_name: results_df, ...}
    output_pdf : Path or str
        Output PDF file path
    """
    # Define all variables to plot (in display order)
    variable_specs = [
        # Economic Variables
        ('y', 'Per-capita output', 'Economic Output Over Time'),
        ('y_eff', 'Effective income ($)', 'Effective Per-Capita Income'),
        ('K', 'Capital stock', 'Capital Stock Over Time'),
        ('C', 'Consumption', 'Consumption Over Time'),
        ('I', 'Investment', 'Investment Over Time'),
        ('s_interp', 'Savings rate', 'Savings Rate Over Time'),

        # Climate Variables
        ('T_atm', 'Atmospheric temp (°C)', 'Atmospheric Temperature'),
        ('T_ocean', 'Ocean temp (°C)', 'Ocean Temperature'),
        ('delta_T', 'Temp change (°C)', 'Temperature Change from Pre-Industrial'),
        ('E', 'Emissions (GtC/yr)', 'CO₂ Emissions Rate'),
        ('E_cum', 'Cumulative emissions (GtC)', 'Cumulative CO₂ Emissions'),
        ('M_atm', 'Atmospheric CO₂ (GtC)', 'Atmospheric CO₂ Concentration'),
        ('M_upper', 'Upper ocean C (GtC)', 'Upper Ocean Carbon'),
        ('M_lower', 'Lower ocean C (GtC)', 'Lower Ocean Carbon'),

        # Abatement and Damage
        ('f_interp', 'Abatement fraction', 'Emissions Abatement Fraction'),
        ('Lambda', 'Abatement cost fraction', 'Abatement Cost (% of GDP)'),
        ('Omega', 'Climate damage fraction', 'Climate Damage (% of GDP)'),
        ('Omega_1', 'Damage (rich)', 'Climate Damage to Rich Group'),
        ('Omega_2', 'Damage (poor)', 'Climate Damage to Poor Group'),

        # Inequality and Utility
        ('G', 'Gini coefficient', 'Gini Coefficient Over Time'),
        ('G_eff', 'Effective Gini', 'Effective Gini Index'),
        ('U', 'Mean utility', 'Mean Utility Over Time'),
        ('U_1', 'Utility (rich)', 'Utility of Rich Group'),
        ('U_2', 'Utility (poor)', 'Utility of Poor Group'),

        # Other
        ('f', 'Abatement allocation', 'Abatement Allocation Fraction'),
    ]

    with PdfPages(output_pdf) as pdf:
        # Process plots in groups of 6
        for page_start in range(0, len(variable_specs), 6):
            page_vars = variable_specs[page_start:page_start + 6]

            # Create 2×3 subplot grid in 16:9 landscape orientation
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            fig.suptitle('Results Comparison', fontsize=14, fontweight='bold')
            axes = axes.flatten()

            for idx, (var_name, ylabel, title) in enumerate(page_vars):
                # Check if variable exists in at least one dataset
                if any(var_name in df.columns for df in case_data.values()):
                    plot_timeseries(axes[idx], case_data, var_name, ylabel, title)
                else:
                    # Variable not found - leave blank or show message
                    axes[idx].text(0.5, 0.5, f'{var_name}\nnot available',
                                  ha='center', va='center',
                                  transform=axes[idx].transAxes)
                    axes[idx].set_title(title, fontsize=11)

            # Hide any unused subplots on last page
            for idx in range(len(page_vars), 6):
                axes[idx].axis('off')

            plt.tight_layout()
            pdf.savefig(fig, orientation='landscape')
            plt.close(fig)

    print(f"Results report saved to: {output_pdf}")
```

### 6. Refactoring Visualization Code

**Extract from test_optimization.py:**

Move `create_visualization_plots()` and related functions to `visualization_utils.py`.

**New module:** `visualization_utils.py`

```python
"""
Visualization utilities for COIN_equality optimization results.

Provides reusable plotting functions for:
- Single-run result visualization (time series plots)
- Multi-run comparison plots
- Summary statistics plots
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from pathlib import Path


def create_timeseries_plot(df, x_col, y_col, xlabel, ylabel, title):
    """
    Create a single time series plot.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing x and y columns
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df[x_col], df[y_col], linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_single_run_report(results_csv, output_pdf):
    """
    Create full visualization report for a single optimization run.

    This is the refactored version of create_visualization_plots().

    Parameters
    ----------
    results_csv : Path or str
        Path to results.csv file
    output_pdf : Path or str
        Output PDF file path
    """
    df = pd.read_csv(results_csv)

    with PdfPages(output_pdf) as pdf:
        # Page 1: Per-capita output
        fig = create_timeseries_plot(
            df, 't', 'y',
            'Time (years)',
            'Per-capita output',
            'Economic Output Over Time'
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Capital stock
        fig = create_timeseries_plot(
            df, 't', 'K',
            'Time (years)',
            'Capital stock',
            'Capital Stock Over Time'
        )
        pdf.savefig(fig)
        plt.close(fig)

        # ... (continue for all variables)


def create_comparison_report(case_results, output_pdf):
    """
    Create comparative visualization report for multiple runs.

    Parameters
    ----------
    case_results : dict
        {case_name: results_df, ...}
    output_pdf : Path or str
        Output PDF file path
    """
    # Implementation similar to single run but with overlays
    pass
```

**Update test_optimization.py:**

```python
from visualization_utils import create_single_run_report

# In main() function, replace create_visualization_plots() call:
create_single_run_report(
    results_csv=output_dir / 'results.csv',
    output_pdf=output_dir / 'plots_full.pdf'
)
```

## Implementation Steps

### Step 1: Create visualization_utils.py with Unified Plotting

**Action:** Create unified plotting functions that work for both single and multi-run cases

**Key Design:** Single plotting function `plot_timeseries()` that accepts:
- `case_data`: dict with one or more cases `{case_name: df, ...}`
- `variable`: column name to plot
- `ylabel`: Y-axis label
- `title`: Plot title
- Single case → one line
- Multiple cases → multiple lines with legend

**Implementation steps:**

1. Create new file `visualization_utils.py`
2. Implement **unified** `plot_timeseries()` function that handles both cases
3. Create `create_results_report_pdf()` that uses `plot_timeseries()` for all variables
4. **Split current dual-variable panels** into separate single-variable panels
5. Define comprehensive variable specification list (25 variables from results.csv)
6. Add comprehensive docstrings following NumPy style
7. Test with existing results to verify output

**Files created:**
- NEW: `visualization_utils.py`

**Files modified:**
- MODIFIED: `test_optimization.py` (will import and use in later step)

### Step 2: Create comparison_utils.py

**Action:** Implement directory discovery and data loading utilities

```python
"""
Utilities for comparing multiple optimization runs.

Provides functions for:
- Discovering result directories from path patterns
- Loading and validating CSV files
- Generating case names from directory paths
"""

import glob
from pathlib import Path
import pandas as pd


def discover_result_directories(path_patterns):
    """
    Discover valid result directories from path patterns.

    Parameters
    ----------
    path_patterns : list of str
        List of directory paths or glob patterns

    Returns
    -------
    list of Path
        Sorted list of directories containing optimization_summary.csv

    Raises
    ------
    ValueError
        If no valid directories found
    """
    directories = []

    for pattern in path_patterns:
        matches = glob.glob(pattern)
        for match in matches:
            path = Path(match)
            if path.is_dir() and (path / 'optimization_summary.csv').exists():
                directories.append(path)

    if not directories:
        raise ValueError(f"No valid result directories found for patterns: {path_patterns}")

    return sorted(set(directories))


def generate_case_name(directory_path):
    """
    Generate readable case name from directory path.

    Parameters
    ----------
    directory_path : Path
        Directory path

    Returns
    -------
    str
        Case name for display
    """
    return directory_path.name


def load_optimization_summaries(directories):
    """
    Load optimization_summary.csv from multiple directories.

    Parameters
    ----------
    directories : list of Path
        Result directories

    Returns
    -------
    dict
        {case_name: pd.DataFrame, ...}
    """
    data = {}
    for directory in directories:
        case_name = generate_case_name(directory)
        csv_path = directory / 'optimization_summary.csv'
        data[case_name] = pd.read_csv(csv_path)

    return data


def load_results_csvs(directories):
    """
    Load results.csv from multiple directories.

    Parameters
    ----------
    directories : list of Path
        Result directories

    Returns
    -------
    dict
        {case_name: pd.DataFrame, ...}
        Only includes cases where results.csv exists
    """
    data = {}
    for directory in directories:
        case_name = generate_case_name(directory)
        csv_path = directory / 'results.csv'
        if csv_path.exists():
            data[case_name] = pd.read_csv(csv_path)
        else:
            print(f"Warning: results.csv not found in {directory}, skipping results comparison")

    return data
```

**Files created:**
- NEW: `comparison_utils.py`

### Step 3: Implement XLSX Generation

**Action:** Create functions for generating comparison Excel workbook

Add to `comparison_utils.py`:

```python
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter


def create_comparison_xlsx(optimization_data, output_path):
    """
    Create Excel workbook comparing optimization summaries across cases.

    Parameters
    ----------
    optimization_data : dict
        {case_name: optimization_summary_df, ...}
    output_path : Path or str
        Output Excel file path
    """
    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # Remove default sheet

    # Define comparison sheets
    comparison_specs = [
        ('Objective', 'objective', '{:.6e}'),
        ('Evaluations', 'n_evaluations', '{:.0f}'),
        ('Elapsed Time (s)', 'elapsed_time', '{:.2f}'),
        ('Termination Status', 'termination_status', '{}')
    ]

    for sheet_name, column_name, number_format in comparison_specs:
        create_comparison_sheet(wb, sheet_name, column_name, optimization_data, number_format)

    # Auto-size columns
    for sheet in wb.worksheets:
        for column in sheet.columns:
            max_length = 0
            column_letter = get_column_letter(column[0].column)
            for cell in column:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            sheet.column_dimensions[column_letter].width = max_length + 2

    wb.save(output_path)
    print(f"Comparison Excel workbook saved to: {output_path}")


def create_comparison_sheet(wb, sheet_name, column_name, data, number_format):
    """
    Create a single comparison sheet in the workbook.

    Parameters
    ----------
    wb : openpyxl.Workbook
        Workbook to add sheet to
    sheet_name : str
        Name of the sheet
    column_name : str
        Column name from optimization_summary.csv to extract
    data : dict
        {case_name: optimization_summary_df, ...}
    number_format : str
        Format string for values
    """
    ws = wb.create_sheet(sheet_name)

    # Header row
    ws['A1'] = 'Iteration'
    ws['A1'].font = Font(bold=True)
    ws['A1'].fill = PatternFill(start_color='CCCCCC', end_color='CCCCCC', fill_type='solid')

    case_names = list(data.keys())
    for col_idx, case_name in enumerate(case_names, start=2):
        cell = ws.cell(1, col_idx, case_name)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color='CCCCCC', end_color='CCCCCC', fill_type='solid')
        cell.alignment = Alignment(horizontal='center')

    # Data rows
    max_iterations = max(len(df) for df in data.values())

    for row_idx in range(max_iterations):
        # Iteration number
        ws.cell(row_idx + 2, 1, row_idx + 1)

        # Values for each case
        for col_idx, (case_name, df) in enumerate(data.items(), start=2):
            if row_idx < len(df):
                value = df.iloc[row_idx][column_name]
                cell = ws.cell(row_idx + 2, col_idx)

                if isinstance(value, (int, float)):
                    cell.value = value
                    cell.number_format = number_format
                else:
                    cell.value = str(value)
```

**Files modified:**
- MODIFIED: `comparison_utils.py`

### Step 4: Implement PDF Comparison Plots

**Action:** Add comparison plotting functions to visualization_utils.py

```python
def create_objective_scatter_comparison(optimization_data):
    """
    Create scatter plot comparing objective values across cases and iterations.

    Parameters
    ----------
    optimization_data : dict
        {case_name: optimization_summary_df, ...}

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    colors = plt.cm.tab10(range(10))

    case_names = list(optimization_data.keys())
    max_iterations = max(len(df) for df in optimization_data.values())

    # Plot each iteration with different markers
    for iter_idx in range(max_iterations):
        marker = markers[min(iter_idx, len(markers) - 1)]

        x_positions = []
        y_values = []
        labels = []

        for case_idx, (case_name, df) in enumerate(optimization_data.items()):
            if iter_idx < len(df):
                x_positions.append(case_idx)
                y_values.append(df.iloc[iter_idx]['objective'])
                labels.append(case_name)

        ax.scatter(
            x_positions,
            y_values,
            marker=marker,
            s=120,
            label=f'Iteration {iter_idx + 1}',
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

    ax.set_xlabel('Case', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('Objective Comparison Across Cases and Iterations', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(case_names)))
    ax.set_xticklabels(case_names, rotation=45, ha='right')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def create_elapsed_time_comparison(optimization_data):
    """
    Create comparison plot for elapsed time across cases and iterations.

    Similar structure to create_objective_scatter_comparison but for elapsed_time.
    """
    # Implementation similar to objective scatter
    pass


def create_timeseries_overlay(results_data, variable_name, ylabel, title):
    """
    Create overlay time series plot comparing a variable across multiple cases.

    Parameters
    ----------
    results_data : dict
        {case_name: results_df, ...}
    variable_name : str
        Column name in results.csv
    ylabel : str
        Y-axis label
    title : str
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10(range(10))

    for idx, (case_name, df) in enumerate(results_data.items()):
        ax.plot(
            df['t'],
            df[variable_name],
            label=case_name,
            linewidth=2,
            color=colors[idx % 10]
        )

    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def create_comparison_report_pdf(optimization_data, results_data, output_pdf):
    """
    Create comprehensive PDF report comparing multiple optimization runs.

    Uses 6-plots-per-page landscape layout for efficient viewing.

    Parameters
    ----------
    optimization_data : dict
        {case_name: optimization_summary_df, ...}
    results_data : dict
        {case_name: results_df, ...}
    output_pdf : Path or str
        Output PDF file path
    """
    with PdfPages(output_pdf) as pdf:
        # Page 1: Summary plots (2×3 grid, 16:9 landscape)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle('Optimization Summary Comparison', fontsize=14, fontweight='bold')
        axes = axes.flatten()

        # Plot 1: Objective comparison scatter
        create_objective_scatter_on_axes(axes[0], optimization_data)

        # Plot 2: Elapsed time comparison
        create_elapsed_time_scatter_on_axes(axes[1], optimization_data)

        # Plot 3: Evaluations comparison (new)
        create_evaluations_scatter_on_axes(axes[2], optimization_data)

        # Plots 4-6: Reserved for additional summary plots or hide
        for idx in range(3, 6):
            axes[idx].axis('off')

        plt.tight_layout()
        pdf.savefig(fig, orientation='landscape')
        plt.close(fig)

        # Pages 2+: Time series comparisons using unified plotting function
        if results_data:
            create_results_report_pdf(results_data, pdf)

    print(f"Comparison PDF report saved to: {output_pdf}")
```

**Files modified:**
- MODIFIED: `visualization_utils.py`

### Step 5: Create Main CLI Script

**Action:** Create compare_results.py with command-line interface

```python
#!/usr/bin/env python3
"""
Compare optimization results across multiple runs.

Usage:
    python compare_results.py <path1> [path2] [path3]

Arguments:
    path1, path2, path3: Directory paths or glob patterns
                         (e.g., "results/run_*/" or "results/baseline/")

Outputs:
    comparison_summary.xlsx - Multi-sheet Excel workbook with tabular comparisons
    comparison_plots.pdf    - PDF report with comparative visualizations
"""

import sys
from pathlib import Path
from comparison_utils import (
    discover_result_directories,
    load_optimization_summaries,
    load_results_csvs,
    create_comparison_xlsx
)
from visualization_utils import create_comparison_report_pdf


def main():
    """Main entry point for comparison tool."""

    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python compare_results.py <path1> [path2] [path3] [...]")
        print("\nArguments:")
        print("  path1, path2, ...: Directory paths or glob patterns (unlimited)")
        print("\nExamples:")
        print('  python compare_results.py results/baseline/ results/sensitivity/')
        print('  python compare_results.py "results/test_*_000/"')
        print('  python compare_results.py results/run_* results/baseline/')
        sys.exit(1)

    path_patterns = sys.argv[1:]

    print(f"Discovering result directories from patterns: {path_patterns}")

    # Discover directories
    directories = discover_result_directories(path_patterns)

    print(f"\nFound {len(directories)} result directories:")
    for d in directories:
        print(f"  - {d}")

    # Load optimization summaries
    print("\nLoading optimization_summary.csv files...")
    optimization_data = load_optimization_summaries(directories)

    # Load results CSVs (if available)
    print("Loading results.csv files...")
    results_data = load_results_csvs(directories)

    # Generate XLSX comparison
    print("\nGenerating Excel comparison workbook...")
    xlsx_path = Path('comparison_summary.xlsx')
    create_comparison_xlsx(optimization_data, xlsx_path)

    # Generate PDF comparison report
    print("Generating PDF comparison report...")
    pdf_path = Path('comparison_plots.pdf')
    create_comparison_report_pdf(optimization_data, results_data, pdf_path)

    print("\n" + "="*80)
    print("Comparison complete!")
    print(f"  Excel workbook: {xlsx_path.absolute()}")
    print(f"  PDF report:     {pdf_path.absolute()}")
    print("="*80)


if __name__ == '__main__':
    main()
```

**Files created:**
- NEW: `compare_results.py`

**Make executable:**
```bash
chmod +x compare_results.py
```

### Step 6: Testing

**Test Cases:**

1. **Single directory:**
   ```bash
   python compare_results.py results/test_000/
   ```

2. **Multiple specific directories:**
   ```bash
   python compare_results.py results/test_000/ results/test_010/
   ```

3. **Wildcard pattern:**
   ```bash
   python compare_results.py "results/test_*/"
   ```

4. **Mixed patterns:**
   ```bash
   python compare_results.py results/baseline/ "results/sensitivity_*/"
   ```

**Validation:**
- Verify XLSX has correct structure and data
- Verify PDF contains all expected plots
- Check that case names are readable
- Verify handling of missing results.csv
- Test with different numbers of iterations

### Step 7: Documentation

**Update README.md** with comparison tool section:

```markdown
### Comparing Multiple Optimization Runs

The `compare_results.py` tool allows you to compare results across multiple optimization runs.

#### Usage

```bash
python compare_results.py <path1> [path2] [path3] [...]
```

#### Arguments

- `path1`, `path2`, ...: Directory paths or glob patterns pointing to result directories
- Unlimited number of path arguments
- Each path can be:
  - A specific directory: `results/baseline/`
  - A wildcard pattern: `results/test_*_000/`

#### Examples

Compare three specific runs:
```bash
python compare_results.py results/baseline/ results/high_eta/ results/low_eta/
```

Compare all matching runs:
```bash
python compare_results.py "results/sensitivity_*/"
```

#### Outputs

1. **comparison_summary.xlsx**: Multi-sheet Excel workbook
   - Sheet 1: Objective values by iteration
   - Sheet 2: Evaluation counts by iteration
   - Sheet 3: Elapsed times by iteration
   - Sheet 4: Termination statuses by iteration

2. **comparison_plots.pdf**: PDF report with visualizations
   - Page 1: Objective value scatter plot (iterations as symbols, cases on x-axis)
   - Page 2: Elapsed time comparison
   - Pages 3+: Time series overlays for all key variables

#### Requirements

Each result directory must contain:
- `optimization_summary.csv` (required)
- `results.csv` (optional, for time series comparisons)

If `results.csv` is missing from a directory, that case will be excluded from time series plots.
```

## Files Modified Summary

### New Files

1. **comparison_utils.py** - Directory discovery, data loading, XLSX generation
2. **visualization_utils.py** - Refactored plotting utilities
3. **compare_results.py** - Main CLI script

### Modified Files

1. **test_optimization.py** - Import and use refactored visualization functions
2. **README.md** - Add comparison tool documentation

## Dependencies

Required packages (already in requirements or standard library):
- `pandas` - CSV loading and data manipulation
- `matplotlib` - Plotting
- `openpyxl` - Excel file generation
- `glob` - Wildcard path expansion (standard library)
- `pathlib` - Path manipulation (standard library)

No new dependencies required.

## Future Enhancements

### Phase 2: Additional Features

1. **Custom case names**: JSON config file for user-specified labels
   ```json
   {
     "results/test_000/": "Baseline",
     "results/test_010/": "High Elasticity"
   }
   ```

2. **Unlimited cases**: Remove 3-case limit, use scrollable legends

3. **Statistical summaries**: Add sheets with mean/std/min/max across cases

4. **Difference plots**: Show differences from baseline case

5. **Configurable variables**: Config file specifying which variables to plot

6. **Output directory option**: `--output-dir` flag to specify where files are saved

7. **Filtering**: `--min-iterations N` to only include cases with N+ iterations

8. **Table exports**: CSV versions of comparison tables

### Phase 3: Interactive Features

1. **HTML report**: Interactive plots with plotly/bokeh
2. **Dashboard**: Web interface for selecting and comparing runs
3. **Database backend**: Store results in SQLite for faster comparison

## Notes

- Follow CLAUDE.md style guide: fail-fast, no defensive programming
- Use pathlib.Path for all file operations
- No backward compatibility needed
- All imports at top of files
- Use NumPy-style docstrings
- No magic numbers - define constants
