"""
Simple Dispersion Analysis for Newmark Method
==============================================
This script plots multiple numerical solutions vs exact solution at the center point.

Usage:
    python dispersion.py file1.csv file2.csv file3.csv ...

File naming convention:
    - mesh_X.csv (e.g., mesh_16.csv, mesh_32.csv) -> Legend shows "mesh 16", "mesh 32"
    - delta_X.csv (e.g., delta_0.01.csv, delta_0.001.csv) -> Legend shows "Δt = 0.01", "Δt = 0.001"

Examples:
    python dispersion.py mesh_8.csv mesh_16.csv mesh_32.csv mesh_64.csv
    python dispersion.py delta_0.1.csv delta_0.05.csv delta_0.01.csv
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
from typing import List, Tuple


def load_center_point_data(filepath: str) -> dict:
    """Load the center point time series from CSV file."""
    data = np.genfromtxt(filepath, delimiter=',', names=True)
    return {
        'time': data['time'],
        'solution': data['solution'],
        'velocity': data['velocity'] if 'velocity' in data.dtype.names else None,
    }


def extract_legend_label(filename: str) -> str:
    """
    Extract legend label from filename.

    Supports formats:
        - mesh_X.csv -> "mesh X"
        - delta_X.csv -> "Δt = X"
        - n_refine_X.csv -> "n_refine X"
        - Any other format -> filename without extension
    """
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]

    # Try to match mesh_X pattern
    mesh_match = re.match(r'mesh[_\-]?(\d+)', name_without_ext, re.IGNORECASE)
    if mesh_match:
        return f"n_refine :  {mesh_match.group(1)}"

    # Try to match delta_X pattern (including decimal numbers)
    delta_match = re.match(r'delta[_\-]?([\d.]+)', name_without_ext, re.IGNORECASE)
    if delta_match:
        return f"Δt :  {delta_match.group(1)}"

    # Try to match n_refine_X pattern
    nrefine_match = re.match(r'n[_\-]?refine[_\-]?(\d+)', name_without_ext, re.IGNORECASE)
    if nrefine_match:
        return f"n_refine {nrefine_match.group(1)}"

    # Try to match dt_X pattern
    dt_match = re.match(r'dt[_\-]?([\d.]+)', name_without_ext, re.IGNORECASE)
    if dt_match:
        return f"Δt = {dt_match.group(1)}"

    # Fallback: return filename without extension
    return name_without_ext


def get_colors(n: int) -> List[str]:
    """Generate a list of distinct colors for plotting."""
    if n <= 10:
        # Use tab10 colormap for up to 10 colors
        cmap = plt.cm.tab10
        return [cmap(i) for i in range(n)]
    else:
        # Use a continuous colormap for more colors
        cmap = plt.cm.viridis
        return [cmap(i / (n - 1)) for i in range(n)]


def plot_solution_comparison(
        filepaths: List[str],
        test_case: str = "EX2",
        save_path: str = None,
        show_exact: bool = True,
        figsize: Tuple[float, float] = (12, 7)
):
    """
    Plot multiple numerical solutions vs exact solution at center point.

    Parameters:
        filepaths: List of paths to CSV files containing numerical solutions
        test_case: "EX1" or "EX2" for determining exact solution frequency
        save_path: Path to save the figure (optional)
        show_exact: Whether to plot the exact solution
        figsize: Figure size as (width, height)
    """
    # Expected frequency for each test case
    if test_case == "EX1":
        omega_exact = 1.0  # cos(t) → ω = 1
    else:  # EX2
        omega_exact = np.pi / np.sqrt(2)  # cos(π/√2 * t) → ω = π/√2 ≈ 2.22

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get colors for numerical solutions
    colors = get_colors(len(filepaths))

    # Track time range for exact solution
    t_min, t_max = float('inf'), float('-inf')

    # Plot each numerical solution
    for i, filepath in enumerate(filepaths):
        if not os.path.exists(filepath):
            print(f"Warning: File not found, skipping: {filepath}")
            continue

        # Load data
        data = load_center_point_data(filepath)
        time = data['time']
        solution = data['solution']

        # Update time range
        t_min = min(t_min, time.min())
        t_max = max(t_max, time.max())

        # Extract label from filename
        label = extract_legend_label(filepath)

        # Plot numerical solution
        ax.plot(
            time, solution,
            color=colors[i],
            linewidth=1.5,
            label=label
        )

        print(f"Loaded {len(time)} time points from: {filepath} (label: {label})")

    # Plot exact solution
    if show_exact and t_min != float('inf'):
        time_exact = np.linspace(t_min, t_max, 1000)
        u0_at_center = np.sin(np.pi * 1.0 / 2.0) * np.sin(np.pi * 1.0 / 2.0)  # sin(π/2)² = 1
        u_exact = u0_at_center * np.cos(omega_exact * time_exact)

        ax.plot(
            time_exact, u_exact,
            'k--',
            linewidth=2,
            alpha=0.8,
            label='Exact solution'
        )

    # Configure plot
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('u(0, 0, t)', fontsize=12)
    ax.set_title(f'Solution at Center Point (0, 0) - Test Case: {test_case}', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.show()


def plot_error_comparison(
        filepaths: List[str],
        test_case: str = "EX2",
        save_path: str = None,
        figsize: Tuple[float, float] = (12, 7)
):
    """
    Plot the error (numerical - exact) for multiple solutions.

    Parameters:
        filepaths: List of paths to CSV files containing numerical solutions
        test_case: "EX1" or "EX2" for determining exact solution frequency
        save_path: Path to save the figure (optional)
        figsize: Figure size as (width, height)
    """
    # Expected frequency for each test case
    if test_case == "EX1":
        omega_exact = 1.0
    else:  # EX2
        omega_exact = np.pi / np.sqrt(2)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get colors
    colors = get_colors(len(filepaths))

    # Plot error for each numerical solution
    for i, filepath in enumerate(filepaths):
        if not os.path.exists(filepath):
            print(f"Warning: File not found, skipping: {filepath}")
            continue

        # Load data
        data = load_center_point_data(filepath)
        time = data['time']
        solution = data['solution']

        # Compute exact solution at same time points
        u0_at_center = 1.0  # sin(π/2)² = 1
        u_exact = u0_at_center * np.cos(omega_exact * time)

        # Compute error
        error = solution - u_exact

        # Extract label from filename
        label = extract_legend_label(filepath)

        # Plot error
        ax.plot(
            time, error,
            color=colors[i],
            linewidth=1.5,
            label=label
        )

    # Configure plot
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('Error: u_numerical - u_exact', fontsize=12)
    ax.set_title(f'Error at Center Point (0, 0) - Test Case: {test_case}', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nError figure saved to: {save_path}")

    plt.show()


def print_usage():
    """Print usage information."""
    print(__doc__)
    print("\nExamples:")
    print("  python dispersion.py mesh_8.csv mesh_16.csv mesh_32.csv")
    print("  python dispersion.py delta_0.1.csv delta_0.05.csv delta_0.01.csv")
    print("  python dispersion.py --error mesh_8.csv mesh_16.csv  # Plot errors")
    print("  python dispersion.py --case EX1 file1.csv file2.csv  # Use EX1 test case")
    print("  python dispersion.py --no-exact file1.csv file2.csv  # Don't show exact solution")


def main():
    """Main function."""
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Error: No input files provided.")
        print_usage()
        sys.exit(1)

    # Parse options
    filepaths = []
    test_case = "EX2"
    plot_errors = False
    show_exact = True
    save_path = "solution_comparison.png"
    error_save_path = "error_comparison.png"

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg in ['-h', '--help']:
            print_usage()
            sys.exit(0)
        elif arg == '--case':
            if i + 1 < len(sys.argv):
                test_case = sys.argv[i + 1].upper()
                i += 1
            else:
                print("Error: --case requires an argument (EX1 or EX2)")
                sys.exit(1)
        elif arg == '--error':
            plot_errors = True
        elif arg == '--no-exact':
            show_exact = False
        elif arg == '--output' or arg == '-o':
            if i + 1 < len(sys.argv):
                save_path = sys.argv[i + 1]
                i += 1
            else:
                print("Error: --output requires a filename")
                sys.exit(1)
        elif arg.startswith('-'):
            print(f"Warning: Unknown option '{arg}', ignoring")
        else:
            filepaths.append(arg)

        i += 1

    # Check that we have files
    if not filepaths:
        print("Error: No input files provided.")
        print_usage()
        sys.exit(1)

    # Check files exist
    valid_files = []
    for fp in filepaths:
        if os.path.exists(fp):
            valid_files.append(fp)
        else:
            print(f"Warning: File not found: {fp}")

    if not valid_files:
        print("Error: No valid input files found.")
        sys.exit(1)

    print(f"\nProcessing {len(valid_files)} file(s)...")
    print(f"Test case: {test_case}")
    print(f"Show exact solution: {show_exact}")

    # Create solution comparison plot
    plot_solution_comparison(
        valid_files,
        test_case=test_case,
        save_path=save_path,
        show_exact=show_exact
    )

    # Create error plot if requested
    if plot_errors:
        plot_error_comparison(
            valid_files,
            test_case=test_case,
            save_path=error_save_path
        )

    print("\nDone!")


if __name__ == "__main__":
    main()