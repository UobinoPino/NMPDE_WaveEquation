"""
Simple Dispersion Analysis for Newmark Method
==============================================
This script plots the numerical vs exact solution at the center point.

Usage:
    python dispersion_simple.py [path_to_center_point_solution.csv]
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def load_center_point_data(filepath: str) -> dict:
    """Load the center point time series from CSV file."""
    data = np.genfromtxt(filepath, delimiter=',', names=True)
    return {
        'time': data['time'],
        'solution': data['solution'],
        'velocity': data['velocity'],
      #  'acceleration': data['acceleration']
    }


def plot_solution_comparison(data: dict, test_case: str = "EX2", save_path: str = None):
    """
    Plot numerical vs exact solution at center point.
    """
    time = data['time']
    solution = data['solution']

    # Expected frequency for each test case
    if test_case == "EX1":
        omega_exact = 1.0  # cos(t) → ω = 1
    else:  # EX2
        omega_exact = np.pi / np.sqrt(2)  # cos(π/√2 * t) → ω = π/√2 ≈ 2.22

    # Compute exact solution
    u0_at_center = np.sin(np.pi * 1.0 / 2.0) * np.sin(np.pi * 1.0 / 2.0)  # sin(π/2)² = 1
    u_exact = u0_at_center * np.cos(omega_exact * time)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(time, solution, 'b-', linewidth=1.5, label='Numerical solution u(0,0,t)')
    ax.plot(time, u_exact, 'r--', linewidth=1.5, alpha=0.7, label='Exact solution')

    ax.set_xlabel('Time t', fontsize=11)
    ax.set_ylabel('u(0, 0, t)', fontsize=11)
    ax.set_title(f'Solution at Center Point (0, 0) - Test Case: {test_case}', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def main():
    """Main function."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "center_point_solution_theta.csv"

    # Check file exists
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        print("\nUsage: python dispersion_simple.py [path_to_csv]")
        sys.exit(1)

    print(f"Loading data from: {filepath}")

    # Load data
    data = load_center_point_data(filepath)
    print(f"Loaded {len(data['time'])} time points")

    # Test case (change to "EX1" if needed)
    test_case = "EX2"

    # Create plot
    plot_solution_comparison(data, test_case, save_path='solution_comparison.png')

    print("\nDone! Generated: solution_comparison.png")


if __name__ == "__main__":
    main()