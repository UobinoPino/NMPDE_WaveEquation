import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
from collections import defaultdict

def parse_filename(filename):
    """
    Parse filename format: energy_<method>_<refinement>_<dt>_<r>.csv
    Example: energy_Newmark_7_0.01_1.csv
    Returns: (method, refinement, dt, r)
    """
    # Remove .csv extension and 'energy_' prefix
    name = filename.replace('.csv', '').replace('energy_', '')
    
    # Try to match pattern: method_refinement_dt_r
    # Method can be letters, refinement is integer, dt is float, r is integer
    match = re.match(r'([A-Za-z]+)_(\d+)_([\d.]+)_(\d+)', name)
    
    if match:
        method = match.group(1)
        refinement = int(match.group(2))
        dt = float(match.group(3))
        r = int(match.group(4))
        return method, refinement, dt, r
    else:
        print(f"Warning: Could not parse filename {filename}")
        return None, None, None, None

def group_files_by_parameter(filepaths):
    """
    Group files by what parameter varies.
    Returns dict with keys 'dt', 'refinement', 'method', or 'mixed'
    """
    parsed_data = []
    for fp in filepaths:
        filename = os.path.basename(fp)
        method, ref, dt, r = parse_filename(filename)
        if method is not None:
            parsed_data.append({
                'filepath': fp,
                'filename': filename,
                'method': method,
                'refinement': ref,
                'dt': dt,
                'r': r
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
    print("Example: python3 plot_energy.py energy_Newmark_7_0.01_1.csv energy_Newmark_7_0.005_1.csv")
    print("Filename format: energy_<method>_<refinement>_<dt>_<r>.csv")
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

# ============================================================================
# PLOT SETTINGS - Modify these values to customize the plot
# ============================================================================
FIGURE_WIDTH = 13       # Width in inches
FIGURE_HEIGHT = 6       # Height in inches
Y_MIN = 2.36            # Y-axis minimum (None for auto)
Y_MAX = 2.52           # Y-axis maximum (None for auto)
X_MIN = 0.0            # X-axis minimum (None for auto)
X_MAX = 2.0            # X-axis maximum (None for auto)
# ============================================================================

# Sort data for consistent plotting
if param_type == 'dt':
    parsed_data.sort(key=lambda x: x['dt'])
elif param_type == 'refinement':
    parsed_data.sort(key=lambda x: x['refinement'])
elif param_type == 'method':
    parsed_data.sort(key=lambda x: x['method'])

# Get method name for title
method_name = parsed_data[0]['method'] if parsed_data else "Unknown"
output_name = f"energy_{method_name}_comparison.png"

# Create plot
fig, ax = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)), plt.gca()

# Exact energy for EX2: E = pi^2 / 4
E_EXACT_EX2 = np.pi**2 / 4
print(f"Exact energy (EX2): E = π²/4 = {E_EXACT_EX2:.16e}")

# Plot exact energy FIRST (so it appears behind numerical data)
plt.axhline(y=E_EXACT_EX2, color='black', linestyle='--', linewidth=2, 
            label=f'Exact energy = $\\pi^2/4 \\approx {E_EXACT_EX2:.4f}$', zorder=1)

# Store mean energies for annotation
initial_energies = []

for i, data_info in enumerate(parsed_data):
    filepath = data_info['filepath']
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}, skipping...")
        continue
    
    # Create label with N_el = 4^refinement, Delta t, and r from filename
    method = data_info['method']
    ref = data_info['refinement']
    dt = data_info['dt']
    r = data_info['r']
    N_el = 4 ** ref  # Number of elements
    
    label = f"$N_{{el}}={N_el}$, $\\Delta t={dt}$, $r={r}$"
    
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
    
    # Store initial energy E(0) for annotation
    E0 = total_energy[0]
    initial_energies.append({'E0': E0, 'color': color, 'label': label})
    
    # Print statistics
    E_final = total_energy[-1]
    mean_E = np.mean(total_energy)
    relative_drift = (E_final - E0) / E0 * 100
    max_drift = np.max(np.abs((total_energy - E0) / E0)) * 100
    print(f"{label}: E0={E0:.6e}, E_final={E_final:.6e}, Mean={mean_E:.6e}, "
          f"Final drift={relative_drift:.4e}%, Max drift={max_drift:.4e}%")

plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Total Energy', fontsize=14)
plt.title(f'Energy ({method_name} Method)', fontsize=16)
# Legend outside plot area on the right, positioned lower
plt.legend(fontsize=13, loc='lower left', bbox_to_anchor=(1.02, 0.0), framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', labelsize=12)

# Apply custom axis limits if specified
if Y_MIN is not None or Y_MAX is not None:
    plt.ylim(Y_MIN, Y_MAX)
if X_MIN is not None or X_MAX is not None:
    plt.xlim(X_MIN, X_MAX)

# Add more precise Y-axis ticks
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
# Set minor ticks for more precision
ax.yaxis.set_minor_locator(MultipleLocator(0.01))

# Annotate initial energy E(0) values on the right side of the plot (staggered to avoid overlap)
x_annotate = X_MAX if X_MAX is not None else ax.get_xlim()[1]

# Sort annotations by energy value and space them out
all_annotations = []
for i, ie in enumerate(initial_energies):
    all_annotations.append({'value': ie['E0'], 'color': ie['color'], 'label': ie['label']})
all_annotations.append({'value': E_EXACT_EX2, 'color': 'black', 'label': 'Exact'})

# Sort by value
all_annotations.sort(key=lambda x: x['value'])

# Calculate spacing to avoid overlap (minimum distance between annotations)
y_range = (Y_MAX - Y_MIN) if (Y_MAX is not None and Y_MIN is not None) else 0.2
min_spacing = y_range * 0.06  # 6% of y-range

# Adjust positions to avoid overlap, but stay within Y_MIN and Y_MAX
adjusted_positions = []
for i, ann in enumerate(all_annotations):
    pos = ann['value']
    if i > 0 and adjusted_positions:
        # Ensure minimum spacing from previous annotation
        if pos - adjusted_positions[-1] < min_spacing:
            pos = adjusted_positions[-1] + min_spacing
    adjusted_positions.append(pos)

# If last position exceeds Y_MAX, shift all down proportionally
if Y_MAX is not None and adjusted_positions and adjusted_positions[-1] > Y_MAX - 0.01:
    overflow = adjusted_positions[-1] - (Y_MAX - 0.01)
    adjusted_positions = [p - overflow for p in adjusted_positions]

# Draw annotations with arrows connecting to actual values
for i, ann in enumerate(all_annotations):
    actual_y = ann['value']
    display_y = adjusted_positions[i]
    
    # Skip if display position is outside visible range
    if Y_MIN is not None and display_y < Y_MIN:
        continue
        
    if abs(actual_y - display_y) > 0.0005:
        # Draw arrow from annotation to actual line
        ax.annotate(f'{actual_y:.6f}', 
                    xy=(x_annotate * 0.98, actual_y),  # Point to the line
                    xytext=(x_annotate * 1.02, display_y),  # Text position
                    fontsize=11, 
                    color=ann['color'],
                    va='center',
                    arrowprops=dict(arrowstyle='-', color=ann['color'], lw=0.5))
    else:
        ax.annotate(f'{actual_y:.6f}', 
                    xy=(x_annotate, actual_y), 
                    xytext=(5, 0), 
                    textcoords='offset points',
                    fontsize=11, 
                    color=ann['color'],
                    va='center')

# Adjust plot to make room for legend on the right
plt.subplots_adjust(left=0.08, right=0.72)

# Save as PNG
plt.savefig(output_name, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_name}")

plt.show()