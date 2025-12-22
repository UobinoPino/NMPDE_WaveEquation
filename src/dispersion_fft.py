"""
FFT-based Dispersion Analysis for Newmark Method
=================================================
This script analyzes the numerical dispersion by comparing the FFT spectrum
of the numerical solution at the center point with the expected analytical frequency. 

Usage:
    python dispersion_fft_analysis.py [path_to_center_point_solution.csv]
    
If no path is provided, it looks for center_point_solution.csv in the current directory. 

Author: Generated for UobinoPino/NMPDE_WaveEquation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import sys
import os


def load_center_point_data(filepath:  str) -> dict:
    """
    Load the center point time series from CSV file.
    
    Parameters:
        filepath: Path to center_point_solution.csv
        
    Returns:
        Dictionary with 'time', 'solution', 'velocity', 'acceleration' arrays
    """
    data = np.genfromtxt(filepath, delimiter=',', names=True)
    return {
        'time': data['time'],
        'solution': data['solution'],
        'velocity': data['velocity'],
        'acceleration':  data['acceleration']
    }


def compute_fft_analysis(time:  np.ndarray, signal: np.ndarray) -> dict:
    """
    Perform FFT analysis on the time series.
    
    Parameters:
        time: Time array
        signal: Signal values at each time
        
    Returns: 
        Dictionary with FFT results
    """
    N = len(signal)
    dt = time[1] - time[0]  # Assuming uniform time step
    
    # Remove DC component (mean)
    signal_centered = signal - np.mean(signal)
    
    # Apply window function to reduce spectral leakage
    window = np.hanning(N)
    signal_windowed = signal_centered * window
    
    # Compute FFT
    fft_values = fft(signal_windowed)
    frequencies = fftfreq(N, dt)
    
    # Keep only positive frequencies
    pos_mask = frequencies > 0
    freqs_pos = frequencies[pos_mask]
    amplitude = 2.0 * np.abs(fft_values[pos_mask]) / N  # Normalize
    
    # Also compute angular frequencies
    omega = 2 * np.pi * freqs_pos
    
    return {
        'frequencies': freqs_pos,
        'omega': omega,
        'amplitude': amplitude,
        'dt': dt,
        'N':  N,
        'T_total': time[-1] - time[0]
    }


def find_dominant_frequency(fft_result: dict, n_peaks: int = 3) -> dict:
    """
    Find the dominant frequency components in the FFT spectrum.
    
    Parameters:
        fft_result: Dictionary from compute_fft_analysis
        n_peaks: Number of peaks to find
        
    Returns: 
        Dictionary with peak information
    """
    amplitude = fft_result['amplitude']
    omega = fft_result['omega']
    frequencies = fft_result['frequencies']
    
    # Find peaks in the amplitude spectrum
    peaks, properties = find_peaks(amplitude, height=0.01 * np.max(amplitude))
    
    if len(peaks) == 0:
        print("Warning: No clear peaks found in spectrum")
        # Return the maximum
        idx_max = np.argmax(amplitude)
        return {
            'omega_numerical': [omega[idx_max]],
            'freq_numerical': [frequencies[idx_max]],
            'amplitude':  [amplitude[idx_max]]
        }
    
    # Sort by amplitude
    peak_amplitudes = amplitude[peaks]
    sorted_indices = np.argsort(peak_amplitudes)[::-1]
    
    # Get top n peaks
    top_peaks = peaks[sorted_indices[:min(n_peaks, len(peaks))]]
    
    return {
        'omega_numerical': omega[top_peaks],
        'freq_numerical': frequencies[top_peaks],
        'amplitude': amplitude[top_peaks],
        'all_peak_indices': peaks
    }


def plot_dispersion_analysis(data: dict, fft_result: dict, peaks: dict, 
                             test_case: str = "EX2", save_path: str = None):
    """
    Create comprehensive plots for the dispersion analysis. 
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'FFT Dispersion Analysis - Newmark Method\nTest Case: {test_case}', 
                 fontsize=14, fontweight='bold')
    
    time = data['time']
    solution = data['solution']
    
    # Expected frequencies for each test case
    if test_case == "EX1": 
        omega_exact = 1.0  # cos(t) → ω = 1
        label_exact = "ω_exact = 1.0 rad/s"
    else:  # EX2
        omega_exact = np.pi / np.sqrt(2)  # cos(π/√2 * t) → ω = π/√2 ≈ 2.22
        label_exact = f"ω_exact = π/√2 ≈ {omega_exact:.4f} rad/s"
    
    # =========================================
    # Plot 1: Time series
    # =========================================
    ax1 = axes[0, 0]
    ax1.plot(time, solution, 'b-', linewidth=1, label='Numerical solution u(0,0,t)')
    
    # Overlay expected solution (if we know it)
    u0_at_center = np.sin(np.pi * 1.0 / 2.0) * np.sin(np.pi * 1.0 / 2.0)  # sin(π/2)² = 1
    u_exact = u0_at_center * np.cos(omega_exact * time)
    ax1.plot(time, u_exact, 'r--', linewidth=1, alpha=0.7, label='Exact solution')
    
    ax1.set_xlabel('Time t', fontsize=11)
    ax1.set_ylabel('u(0, 0, t)', fontsize=11)
    ax1.set_title('Solution at Center Point (0, 0)', fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # =========================================
    # Plot 2: FFT Amplitude Spectrum
    # =========================================
    ax2 = axes[0, 1]
    omega = fft_result['omega']
    amplitude = fft_result['amplitude']
    
    ax2.plot(omega, amplitude, 'b-', linewidth=1.5, label='FFT amplitude')
    
    # Mark expected frequency
    ax2.axvline(x=omega_exact, color='r', linestyle='--', linewidth=2, 
                label=label_exact)
    
    # Mark detected peaks
    for i, (w, a) in enumerate(zip(peaks['omega_numerical'], peaks['amplitude'])):
        ax2.plot(w, a, 'go', markersize=10)
        ax2.annotate(f'ω = {w:.4f}', xy=(w, a), xytext=(w + 0.1, a * 1.1),
                     fontsize=9, color='green')
    
    ax2.set_xlabel('Angular Frequency ω (rad/s)', fontsize=11)
    ax2.set_ylabel('Amplitude', fontsize=11)
    ax2.set_title('FFT Spectrum', fontsize=11)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, min(10, omega[-1])])  # Focus on low frequencies
    
    # =========================================
    # Plot 3: Zoom on main peak
    # =========================================
    ax3 = axes[1, 0]
    
    # Zoom around the expected frequency
    zoom_range = 0.5  # ±0.5 rad/s around expected
    mask = (omega > omega_exact - zoom_range) & (omega < omega_exact + zoom_range)
    
    if np.any(mask):
        ax3.plot(omega[mask], amplitude[mask], 'b-', linewidth=2)
        ax3.axvline(x=omega_exact, color='r', linestyle='--', linewidth=2, 
                    label=f'Exact:  ω = {omega_exact:.4f}')
        
        # Find numerical peak in this range
        if len(peaks['omega_numerical']) > 0:
            omega_num = peaks['omega_numerical'][0]
            ax3.axvline(x=omega_num, color='g', linestyle='-', linewidth=2, 
                        label=f'Numerical: ω = {omega_num:.4f}')
    
    ax3.set_xlabel('Angular Frequency ω (rad/s)', fontsize=11)
    ax3.set_ylabel('Amplitude', fontsize=11)
    ax3.set_title('Zoom Around Expected Frequency', fontsize=11)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # =========================================
    # Plot 4: Dispersion error analysis
    # =========================================
    ax4 = axes[1, 1]
    
    # Compute phase error over time
    omega_num = peaks['omega_numerical'][0] if len(peaks['omega_numerical']) > 0 else omega_exact
    
    # Phase at each time step
    phase_exact = omega_exact * time
    phase_numerical = omega_num * time
    phase_error = phase_numerical - phase_exact
    
    ax4.plot(time, phase_error, 'b-', linewidth=1.5)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=1)
    
    ax4.set_xlabel('Time t', fontsize=11)
    ax4.set_ylabel('Phase Error (rad)', fontsize=11)
    ax4.set_title('Accumulated Phase Error:  φ_num - φ_exact', fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # Add text box with summary
    relative_error = (omega_num - omega_exact) / omega_exact * 100
    textstr = f'ω_exact = {omega_exact:.6f} rad/s\n'
    textstr += f'ω_numerical = {omega_num:.6f} rad/s\n'
    textstr += f'Relative error = {relative_error:.4f}%\n'
    textstr += f'Δω = {omega_num - omega_exact:.6f} rad/s'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax4.text(0.95, 0.95, textstr, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return omega_exact, omega_num


def print_analysis_report(data: dict, fft_result:  dict, peaks: dict, 
                          test_case: str = "EX2"):
    """
    Print a detailed analysis report.
    """
    # Expected frequencies
    if test_case == "EX1": 
        omega_exact = 1.0
        T_exact = 2 * np.pi
    else: 
        omega_exact = np.pi / np.sqrt(2)
        T_exact = 2 * np.pi / omega_exact
    
    omega_num = peaks['omega_numerical'][0] if len(peaks['omega_numerical']) > 0 else np.nan
    
    print("\n" + "=" * 70)
    print("FFT DISPERSION ANALYSIS REPORT")
    print("=" * 70)
    
    print(f"\n--- Simulation Parameters ---")
    print(f"  Time step Δt:         {fft_result['dt']:.6f} s")
    print(f"  Total time:           {fft_result['T_total']:.2f} s")
    print(f"  Number of samples:   {fft_result['N']}")
    print(f"  Test case:           {test_case}")
    
    print(f"\n--- Expected (Analytical) ---")
    print(f"  Angular frequency:   ω_exact = {omega_exact:.6f} rad/s")
    print(f"  Period:               T_exact = {T_exact:.6f} s")
    print(f"  Frequency:           f_exact = {omega_exact / (2*np.pi):.6f} Hz")
    
    print(f"\n--- Numerical (from FFT) ---")
    print(f"  Angular frequency:   ω_num = {omega_num:.6f} rad/s")
    if not np.isnan(omega_num):
        T_num = 2 * np.pi / omega_num
        print(f"  Period:              T_num = {T_num:.6f} s")
        print(f"  Frequency:           f_num = {omega_num / (2*np.pi):.6f} Hz")
    
    print(f"\n--- Dispersion Error ---")
    if not np.isnan(omega_num):
        abs_error = omega_num - omega_exact
        rel_error = abs_error / omega_exact * 100
        period_error = (T_num - T_exact) / T_exact * 100
        
        print(f"  Δω = ω_num - ω_exact = {abs_error:.8f} rad/s")
        print(f"  Relative freq error:   {rel_error:.6f}%")
        print(f"  Period elongation:    {period_error:.6f}%")
        
        if abs_error < 0:
            print(f"\n  → Numerical wave is SLOWER (ω_num < ω_exact)")
            print(f"    This is typical 'lagging' dispersion for implicit methods")
        else:
            print(f"\n  → Numerical wave is FASTER (ω_num > ω_exact)")
    
    # Check Ω = ω·Δt (should be small for accurate time integration)
    Omega = omega_exact * fft_result['dt']
    print(f"\n--- Time Integration Quality ---")
    print(f"  Ω = ω·Δt = {Omega:.6f}")
    if Omega < 0.1:
        print(f"  ✓ Excellent (Ω < 0.1): very small time integration error expected")
    elif Omega < 0.3:
        print(f"  ✓ Good (Ω < 0.3): small time integration error")
    elif Omega < 0.5:
        print(f"  ⚠ Moderate (Ω < 0.5): noticeable dispersion expected")
    else:
        print(f"  ✗ Large (Ω > 0.5): significant dispersion, consider smaller Δt")
    
    print("\n" + "=" * 70)


def main():
    """Main function."""
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "center_point_solution.csv"
    
    # Check file exists
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        print("\nUsage: python dispersion_fft_analysis.py [path_to_csv]")
        print("\nMake sure you have run the Newmark simulation first,")
        print("which should generate 'center_point_solution.csv'")
        sys.exit(1)
    
    print(f"Loading data from: {filepath}")
    
    # Load data
    data = load_center_point_data(filepath)
    print(f"Loaded {len(data['time'])} time points")
    
    # Detect test case from initial value
    # At (0,0) with u0 = sin(π(x+1)/2)*sin(π(y+1)/2), we have u0(0,0) = sin(π/2)*sin(π/2) = 1
    u0 = data['solution'][0]
    print(f"Initial value u(0,0,0) = {u0:.6f}")
    
    # Try to detect test case (EX1 has ω=1, EX2 has ω=π/√2)
    # We'll default to EX2 since that's what's in your main file
    test_case = "EX2"  # Change to "EX1" if needed
    
    # Perform FFT analysis
    print("\nComputing FFT...")
    fft_result = compute_fft_analysis(data['time'], data['solution'])
    
    # Find dominant frequencies
    peaks = find_dominant_frequency(fft_result)
    print(f"Dominant frequency found: ω = {peaks['omega_numerical'][0]:.6f} rad/s")
    
    # Print detailed report
    print_analysis_report(data, fft_result, peaks, test_case)
    
    # Create plots
    print("\nGenerating plots...")
    plot_dispersion_analysis(data, fft_result, peaks, test_case, 
                            save_path='dispersion_fft_analysis.png')
    
    # Also plot velocity and acceleration spectra
    print("\nAnalyzing velocity spectrum...")
    fft_velocity = compute_fft_analysis(data['time'], data['velocity'])
    peaks_velocity = find_dominant_frequency(fft_velocity)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fft_velocity['omega'], fft_velocity['amplitude'], 'b-', linewidth=1.5)
    omega_exact = 1.0 if test_case == "EX1" else np.pi / np.sqrt(2)
    ax.axvline(x=omega_exact, color='r', linestyle='--', linewidth=2, 
               label=f'Expected: ω = {omega_exact:.4f}')
    ax.set_xlabel('Angular Frequency ω (rad/s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('FFT of Velocity at Center Point')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 10])
    plt.tight_layout()
    plt.savefig('velocity_fft.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nAnalysis complete!")
    print("Generated files:")
    print("  - dispersion_fft_analysis.png")
    print("  - velocity_fft.png")


if __name__ == "__main__": 
    main()