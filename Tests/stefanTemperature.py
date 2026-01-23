from Scripts import stefan_simulation
import numpy as np
import matplotlib.pyplot as plt

def formfunction(x, shape):
    """
    Defines the shape of north boundary
    takes an array and shape
    returns an array
    """
    if x.size == 0:
        return np.array([])
        
    h1 = x[-1]           # west boundary height (based on total length)
    h2 = x[-1] / 10 * 4  # east boundary height
    l  = x[-1]           # domain length
    
    if shape == 'linear':
        m = (h2 - h1) / (2 * l)
        b = h1 / 2
        return m * x + b
    
    elif shape == 'rectangular':
        return l * np.ones((x.size))
        
    elif shape == 'quadratic':
        # Avoid division by zero if h1 == h2
        if h1 == h2:
            return l * np.ones((x.size))
        k = 2 * l**2 / (h1 - h2)
        return (x - l)**2 / k + h2 / 2 
    
    elif shape == 'crazy':
        return h1/2 + (h2/2 - h1/2) * x / l + 0.25*(-h1 + h2/2) * np.sin(np.pi*x/l)**2
    
    else:
        raise ValueError('Unknown shape: %s' % shape)

def setUpMesh(nodes_x, nodes_y, length, formfunction, shape):
    """
    Generates a mesh where every cell has approximately the same area.
    This is achieved by clustering x-nodes in taller regions.
    """
    # 1. Create a high-resolution temporary grid to measure the area profile
    # We use 10,000 points to ensure the integral is accurate
    x_fine = np.linspace(0, length, 10000)
    y_fine = formfunction(x_fine, shape)
    
    # 2. Calculate the Cumulative Area (Integral of height dx)
    # Using trapezoidal integration
    dx_fine = length / (len(x_fine) - 1)
    # Area of each tiny slice
    slice_areas = (y_fine[:-1] + y_fine[1:]) / 2 * dx_fine
    # Cumulative sum [0, area_1, area_1+area_2, ...]
    cum_area = np.concatenate(([0], np.cumsum(slice_areas)))
    
    # 3. Determine target areas for our mesh nodes
    # We want 'nodes_x' points equally spaced in terms of Area, not Length
    total_area = cum_area[-1]
    target_areas = np.linspace(0, total_area, nodes_x)
    
    # 4. Inverse Interpolation
    # Find the x-coordinates that correspond to these target areas
    # np.interp(target_y, known_y, known_x)
    X_nodes_1D = np.interp(target_areas, cum_area, x_fine)
    
    # 5. Generate the final 2D Mesh
    # Recalculate the exact boundary height at our new, non-uniform X nodes
    north_shape = formfunction(X_nodes_1D, shape)
    
    # Create 2D X matrix (repeat the 1D array for every row)
    X = np.tile(X_nodes_1D, (nodes_y, 1))
    
    # Create 2D Y matrix
    # For each column, space points linearly from Top (north_shape) to Bottom (0)
    Y = np.zeros((nodes_y, nodes_x))
    for j in range(nodes_x):
        Y[:, j] = np.linspace(north_shape[j], 0, nodes_y)
        
    return X, Y[::-1]


shape =  'rectangular'    # 'rectangular', 'linear',  'quadratic', 'crazy'

l = 0.1

# Example usage
Lx, Ly = 0.1, 0.1
dimX, dimY = 4, 24
X, Y = setUpMesh(dimX, dimY, l, formfunction, shape)
initial_temp = np.ones((dimY, dimX)) * 273.15 # Initial temperature field (in Kelvin)
initial_temp[int(dimY/2):, :] += 0.1
x = np.linspace(220, 273, int(dimY/2))[:, None]
initial_temp[:int(dimY/2), :] = x
fl_field_init = np.ones((dimY, dimX))
fl_field_init[:int(dimY/2),:] = 0.0
    
time_step = 0.01  # seconds
steps_no = 20000    # number of time steps to simulate

simulation = stefan_simulation.StefanSimulation(X, Y, initial_temp, time_step, steps_no, q=[-2000, 0, 0, 0], fl_field_init=fl_field_init)
simulation.run()

fig, ax1 = plt.subplots(figsize=(6, 4))

# Primary axis for temperature
ax1.plot(simulation.timeHistory, simulation.THistory[:, int(dimY/2+1), 1], label='Center Point Temperature', color='tab:blue')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Temperature (K)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid()

# Secondary axis for liquid fraction
ax2 = ax1.twinx()
ax2.plot(simulation.timeHistory, simulation.flHistory[:, int(dimY/2+1), 1], label='Center Point Liquid Fraction', color='tab:orange')
ax2.set_ylabel('Liquid Fraction', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.show()

# Temperature domain plot - Temperature vs Depth for 10 different time steps
fig, ax = plt.subplots(figsize=(10, 6))

# Select 10 evenly spaced time steps
num_steps = len(simulation.timeHistory)
step_indices = np.linspace(0, num_steps - 1, 10, dtype=int)

# Get depth values (Y coordinates at center point, x index = 1)
center_x_idx = 1
depths = Y[:, center_x_idx]

# Plot temperature profile for each time step
colors = plt.cm.viridis(np.linspace(0, 1, 10))

for idx, (step_idx, color) in enumerate(zip(step_indices, colors)):
    # Extract temperature field at this time step and center x location
    T_profile = simulation.THistory[step_idx, :, center_x_idx]
    time_value = simulation.timeHistory[step_idx]
    
    ax.plot(depths, T_profile, label=f't={time_value:.1f}s', 
            color=color, linewidth=2, marker='o', markersize=4, alpha=0.8)

ax.set_xlabel('Depth (m)', fontweight='bold', fontsize=12)
ax.set_ylabel('Temperature (K)', fontweight='bold', fontsize=12)
ax.set_title('Temperature Profile vs Depth at Different Time Steps', fontweight='bold', fontsize=13)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.show()

print(T_profile)
# Liquid Fraction domain plot - Liquid Fraction vs Depth for 10 different time steps
fig, ax = plt.subplots(figsize=(10, 6))

# Use same 10 evenly spaced time steps
colors = plt.cm.viridis(np.linspace(0, 1, 10))

for idx, (step_idx, color) in enumerate(zip(step_indices, colors)):
    # Extract liquid fraction field at this time step and center x location
    fl_profile = simulation.flHistory[step_idx, :, center_x_idx]
    time_value = simulation.timeHistory[step_idx]
    
    ax.plot(depths, fl_profile, label=f't={time_value:.1f}s', 
            color=color, linewidth=2, marker='s', markersize=4, alpha=0.8)

ax.set_xlabel('Depth (m)', fontweight='bold', fontsize=12)
ax.set_ylabel('Liquid Fraction', fontweight='bold', fontsize=12)
ax.set_title('Liquid Fraction Profile vs Depth at Different Time Steps', fontweight='bold', fontsize=13)
ax.set_ylim([-0.05, 1.05])
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.show()

