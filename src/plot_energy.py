import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re

# Check command line arguments
if len(sys.argv) < 2:
    print("Usage: python3 plot_energy.py <file1.csv> [file2.csv] [file3.csv] ...")
    print("Example (Newmark): python3 plot_energy.py EX1_energy_0.5_0.25.csv EX1_energy_0.6_0.3025.csv")
    print("Example (Theta):   python3 plot_energy.py EX1_energy_0.5.csv EX1_energy_1.0.csv")
    print("Filename format Newmark: EX<n>_energy_<gamma>_<beta>.csv")
    print("Filename format Theta:   EX<n>_energy_<theta>.csv")
    sys.exit(1)

# Get file paths from command line
filepaths = sys.argv[1:]

# Colors for different curves
colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink', 'gray']

# Fixed y-axis scale for comparability
Y_MIN = 2.20
Y_MAX = 2.550  # Adjust based on your typical energy values

# Detect method type from first file
def detect_method_and_parse(filename):
    """
    Detect if file is Newmark (2 parameters) or Theta (1 parameter) format.
    Expected formats:
      - Newmark: EX<n>_energy_<gamma>_<beta>.csv
      - Theta:   EX<n>_energy_<theta>.csv
    Returns: (method_type, ex_number, label)
    """
    # Pattern for Newmark: EX<n>_energy_<gamma>_<beta>.csv
    newmark_pattern = r'^EX(\d+)_energy_(\d+\.?\d*)_(\d+\.?\d*)\.csv$'
    # Pattern for Theta: EX<n>_energy_<theta>.csv
    theta_pattern = r'^EX(\d+)_energy_(\d+\.?\d*)\.csv$'
    
    newmark_match = re.match(newmark_pattern, filename)
    theta_match = re.match(theta_pattern, filename)
    
    if newmark_match:
        ex_num = newmark_match.group(1)
        gamma = newmark_match.group(2)
        beta = newmark_match.group(3)
        label = rf'$\gamma={gamma}, \beta={beta}$'
        return 'newmark', ex_num, label
    elif theta_match:
        ex_num = theta_match.group(1)
        theta = theta_match.group(2)
        label = rf'$\theta={theta}$'
        return 'theta', ex_num, label
    else:
        # Fallback: use filename as label
        label = filename.replace('.csv', '')
        return 'unknown', None, label

# Determine method type and exercise number from first valid file
method_type = None
ex_number = None
for filepath in filepaths:
    if os.path.exists(filepath):
        filename = os.path.basename(filepath)
        method_type, ex_number, _ = detect_method_and_parse(filename)
        if method_type != 'unknown':
            break

if method_type is None or method_type == 'unknown':
    print("Error: No valid files found or invalid filename format")
    print("Expected formats:")
    print("  Newmark: EX<n>_energy_<gamma>_<beta>.csv")
    print("  Theta:   EX<n>_energy_<theta>.csv")
    sys.exit(1)

# Create plot
plt.figure(figsize=(10, 6))

for i, filepath in enumerate(filepaths):
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}, skipping...")
        continue
    
    filename = os.path.basename(filepath)
    _, _, label = detect_method_and_parse(filename)
    
    # Load data
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    time = data[:, 0]
    total_energy = data[:, 1]
    
    # Plot
    color = colors[i % len(colors)]
    plt.plot(time, total_energy, color=color, linewidth=1.5, label=label)
    
    # Print statistics
    E0 = total_energy[0]
    E_final = total_energy[-1]
    max_drift = np.max(np.abs((total_energy - E0) / E0)) * 100
    print(f"{label}: E0={E0:.4e}, E_final={E_final:.4e}, Max drift={max_drift:.20e}%")

plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Total Energy', fontsize=14)

# Set title based on method type and exercise number
if method_type == 'newmark':
    plt.title(f'Energy (Newmark Method)', fontsize=16)
elif method_type == 'theta':
    plt.title(f'Energy (Theta Method)', fontsize=16)
else:
    plt.title('Energy', fontsize=16)

# Larger, more readable legend
plt.legend(fontsize=16, loc='best', framealpha=0.9)

# Fixed y-axis scale for comparability between plots
plt.ylim(Y_MIN, Y_MAX)

plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', labelsize=12)

# Save as PNG in the current directory
output_path = 'energy_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"\nPlot saved to: {output_path}")
plt.show()