import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re

# Check command line arguments
if len(sys.argv) < 2:
    print("Usage: python3 plot_energy. py <file1.csv> [file2.csv] [file3.csv] ...")
    print("Example: python3 plot_energy.py energy_0.5_0.25.csv energy_0.6_0.3025.csv")
    print("Filename format: energy_<gamma>_<beta>.csv")
    sys.exit(1)

# Get file paths from command line
filepaths = sys. argv[1:]

# Colors for different curves
colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink', 'gray']

# Create plot
plt.figure(figsize=(10, 6))

for i, filepath in enumerate(filepaths):
    # Check if file exists
    if not os.path. exists(filepath):
        print(f"Warning: File not found: {filepath}, skipping...")
        continue
    
    # Extract gamma and beta from filename
    # Expected format: energy_<gamma>_<beta>. csv or any name with two numbers
    filename = os.path.basename(filepath)
    
    # Try to extract gamma and beta from filename
    # Matches patterns like: energy_0.5_0.25.csv, energy_gamma_0.6_beta_0.3025.csv, etc.
    numbers = re.findall(r'(\d+\. ?\d*)', filename)
    
    if len(numbers) >= 2:
        gamma = numbers[0]
        beta = numbers[1]
        label = rf'$\gamma={gamma}, \beta={beta}$'
    else:
        # Fallback:  use filename as label
        label = filename. replace('.csv', '')
    
    # Load data
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    time = data[: , 0]
    total_energy = data[:, 1]
    
    # Plot
    color = colors[i % len(colors)]
    plt.plot(time, total_energy, color=color, linewidth=1.5, label=label)
    
    # Print statistics
    E0 = total_energy[0]
    E_final = total_energy[-1]
    max_drift = np.max(np.abs((total_energy - E0) / E0)) * 100
    print(f"{label}:  E0={E0:.4e}, E_final={E_final:.4e}, Max drift={max_drift:.2e}%")

plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Total Energy', fontsize=12)
plt.title('Energy (Newmark Method)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Save as PNG in the current directory
output_path = 'energy_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"\nPlot saved to: {output_path}")
plt.show()