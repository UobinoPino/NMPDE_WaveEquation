import numpy as np
import matplotlib. pyplot as plt
import sys
import os

# Check command line arguments
if len(sys.argv) < 2:
    print("Usage: python3 plot_errors. py <errors.csv>")
    print("Example: python3 plot_errors.py build/errors.csv")
    sys.exit(1)

# Get file path from command line
filepath = sys.argv[1]

# Check if file exists
if not os.path. exists(filepath):
    print(f"Error: File not found:  {filepath}")
    sys.exit(1)

# Load data
data = np.loadtxt(filepath, delimiter=',', skiprows=1)
time = data[: , 0]
L2_error = data[:, 1]
H1_error = data[:, 2]

# Create single plot
plt.figure(figsize=(10, 6))

# Plot both errors with different colors
plt.plot(time, L2_error, color='blue', linewidth=1.5, label='$L^2$ Error')
plt.plot(time, H1_error, color='red', linewidth=1.5, label='$H^1$ Error')

# Configure plot
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.title('Error Norms vs Time (Newmark Method)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale often useful for errors

# Print statistics
print(f"\n=== Error Statistics ===")
print(f"L2 error: min={np.min(L2_error):.4e}, max={np.max(L2_error):.4e}, final={L2_error[-1]:.4e}")
print(f"H1 error: min={np. min(H1_error):.4e}, max={np. max(H1_error):.4e}, final={H1_error[-1]:.4e}")

# Save as PNG in the same folder as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, 'errors_plot.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"\nPlot saved to: {output_path}")
plt.show()