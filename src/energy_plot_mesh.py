import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
from collections import defaultdict

def parse_filename(filename):
    """
    Parse filename format: energy_<method>_<refinement>_<dt>.csv
    Example: energy_Newmark_7_0.01.csv
    Returns: (method, refinement, dt)
    """
    # Remove .csv extension and 'energy_' prefix
    name = filename.replace('.csv', '').replace('energy_', '')
    
    # Try to match pattern: method_refinement_dt
    # Method can be letters, refinement is integer, dt is float
    match = re.match(r'([A-Za-z]+)_(\d+)_([\d.]+)', name)
    
    if match:
        method = match.group(1)
        refinement = int(match.group(2))
        dt = float(match.group(3))
        return method, refinement, dt
    else:
        print(f"Warning: Could not parse filename {filename}")
        return None, None, None

def group_files_by_parameter(filepaths):
    """
    Group files by what parameter varies.
    Returns dict with keys 'dt', 'refinement', 'method', or 'mixed'
    """
    parsed_data = []
    for fp in filepaths:
        filename = os.path.basename(fp)
        method, ref, dt = parse_filename(filename)
        if method is not None:
            parsed_data.append({
                'filepath': fp,
                'filename': filename,
                'method': method,
                'refinement': ref,
                'dt': dt
            })
    
    if not parsed_data:
        return None, parsed_data
    
    # Check what varies
    methods = set(d['method'] for d in parsed_data)
    refinements = set(d['refinement'] for d in parsed_data)
    dts = set(d['dt'] for d in parsed_data)
    
    if len(dts) > 1 and len(refinements) == 1 and len(methods) == 1:
        return 'dt', parsed_data
    elif len(refinements) > 1 and len(dts) == 1 and len(methods) == 1:
        return 'refinement', parsed_data
    elif len(methods) > 1 and len(dts) == 1 and len(refinements) == 1:
        return 'method', parsed_data
    else:
        return 'mixed', parsed_data

# Check command line arguments
if len(sys.argv) < 2:
    print("Usage: python3 plot_energy.py <file1.csv> [file2.csv] [file3.csv] ...")
    print("Example: python3 plot_energy.py energy_Newmark_7_0.01.csv energy_Newmark_7_0.005.csv")
    print("Filename format: energy_<method>_<refinement>_<dt>.csv")
    print("\nThe script will automatically detect what parameter varies and create appropriate plots.")
    sys.exit(1)

# Get file paths from command line
filepaths = sys.argv[1:]

# Detect what parameter varies
param_type, parsed_data = group_files_by_parameter(filepaths)

if param_type is None:
    print("Error: Could not parse any filenames. Please check filename format.")
    sys.exit(1)

# Colors for different curves
colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

# Sort data for consistent plotting
if param_type == 'dt':
    parsed_data.sort(key=lambda x: x['dt'])
    title_suffix = f"varying $\\Delta t$ (refinement={parsed_data[0]['refinement']})"
    output_name = f"energy_{parsed_data[0]['method']}_varying_dt.png"
elif param_type == 'refinement':
    parsed_data.sort(key=lambda x: x['refinement'])
    title_suffix = f"varying mesh refinement ($\\Delta t={parsed_data[0]['dt']}$)"
    output_name = f"energy_{parsed_data[0]['method']}_varying_mesh.png"
elif param_type == 'method':
    parsed_data.sort(key=lambda x: x['method'])
    title_suffix = f"comparing methods (ref={parsed_data[0]['refinement']}, $\\Delta t={parsed_data[0]['dt']}$)"
    output_name = "energy_method_comparison.png"
else:
    title_suffix = "mixed parameters"
    output_name = "energy_comparison.png"

# Create plot
plt.figure(figsize=(10, 6))

for i, data_info in enumerate(parsed_data):
    filepath = data_info['filepath']
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}, skipping...")
        continue
    
    # Create label based on what varies
    method = data_info['method']
    ref = data_info['refinement']
    dt = data_info['dt']
    
    if param_type == 'dt':
        label = f"$\\Delta t = {dt}$"
    elif param_type == 'refinement':
        label = f"refinement = {ref}"
    elif param_type == 'method':
        label = f"{method}"
    else:
        label = f"{method}, ref={ref}, $\\Delta t={dt}$"
    
    # Load data
    try:
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        time = data[:, 0]
        total_energy = data[:, 1]
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        continue
    
    # Plot
    color = colors[i % len(colors)]
    plt.plot(time, total_energy, color=color, linewidth=1.5, label=label)
    
    # Print statistics
    E0 = total_energy[0]
    E_final = total_energy[-1]
    relative_drift = (E_final - E0) / E0 * 100
    max_drift = np.max(np.abs((total_energy - E0) / E0)) * 100
    print(f"{label}: E0={E0:.6e}, E_final={E_final:.6e}, "
          f"Final drift={relative_drift:.4e}%, Max drift={max_drift:.4e}%")

plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Total Energy', fontsize=12)
plt.title(f'Energy Evolution: {title_suffix}', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Save as PNG
plt.savefig(output_name, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_name}")

plt.show()