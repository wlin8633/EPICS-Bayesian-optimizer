import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set file path (fill in the name of the CSV downloaded after simulation)
CSV_PATH = os.path.join(os.path.dirname(__file__), 'CBO.csv')

# Load data (based on previous preview, the file seems to be tab-separated)
# If loading fails, change sep='\t' to sep=','
print(f"Loading data: {CSV_PATH}")
df = pd.read_csv(CSV_PATH, sep='\t')

# Ensure data is sorted by shot-number
df = df.sort_values('shot-number').reset_index(drop=True)

# Define parameter names for analysis
params = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
target = 'obj'

# Create a directory to store output plots
plot_dir = os.path.join(os.path.dirname(__file__), 'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# ==========================================
# 1. Optimization Trace Plot
# ==========================================
print("Plotting 1. Optimization Trace...")
plt.figure(figsize=(10, 6))

# Calculate current best score over time
df['best_obj'] = df[target].cummax()

# Plot actual score for each shot (scatter points)
plt.scatter(df['shot-number'], df[target], color='gray', alpha=0.5, label='Measured Obj')
# Plot current best score curve (solid line)
plt.plot(df['shot-number'], df['best_obj'], color='red', linewidth=2, label='Current Best $f^+$')

plt.title('Bayesian Optimization Trace')
plt.xlabel('Shot Number (Iteration)')
plt.ylabel('Objective Value')
plt.yscale('log') # Depending on data, target values may vary greatly; log scale might look better. Comment out if not needed.
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '1_optimization_trace.png'))
plt.show(block=False)

# ==========================================
# 2. Parallel Coordinates Plot
# ==========================================
print("Plotting 2. Parallel Coordinates...")
fig, ax = plt.subplots(figsize=(12, 6))

# Normalize 7 parameters to [0, 1] range so they can be shown on the same Y-axis
df_norm = df.copy()
for p in params:
    min_val, max_val = df[p].min(), df[p].max()
    if max_val > min_val:
        df_norm[p] = (df[p] - min_val) / (max_val - min_val)
    else:
        df_norm[p] = 0.5

# Set color mapping (warmer/brighter color for higher score)
cmap = plt.get_cmap('plasma')
norm = plt.Normalize(df[target].min(), df[target].max())

x_coords = np.arange(len(params))
for _, row in df_norm.iterrows():
    y_coords = row[params].values
    color = cmap(norm(row[target]))
    # Lower transparency for lower scores, solid for higher scores to highlight the trend
    alpha = 0.2 if row[target] < df['best_obj'].quantile(0.8) else 0.8
    ax.plot(x_coords, y_coords, color=color, alpha=alpha, linewidth=1.5)

ax.set_xticks(x_coords)
ax.set_xticklabels(params)
ax.set_ylabel("Normalized Parameter Value [0, 1]")
ax.set_title("Parallel Coordinates (Colored by Objective)")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(sm, ax=ax, label='Objective Value')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '2_parallel_coordinates.png'))
plt.show(block=False)

# ==========================================
# 3. Parameter Trajectory over Time
# ==========================================
print("Plotting 3. Parameter Trajectory over Time...")
fig, axes = plt.subplots(7, 1, figsize=(10, 12), sharex=True)
fig.suptitle('Parameter Exploration Trajectory', fontsize=14)

for i, p in enumerate(params):
    # Use scatter plot, color represents Objective score for that point
    sc = axes[i].scatter(df['shot-number'], df[p], c=df[target], cmap='plasma', alpha=0.7, s=20)
    axes[i].set_ylabel(p)
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel('Shot Number')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '3_parameter_trajectory.png'))
plt.show(block=False)

# ==========================================
# 4. Top 3 Important Parameters (2D Scatter Matrix)
# ==========================================
print("Plotting 4. Top 3 Important Parameters...")
# Calculate absolute Pearson correlation with the target to find the top 3
correlations = df[params + [target]].corr()[target].abs().drop(target)
top_3_params = correlations.sort_values(ascending=False).head(3).index.tolist()
print(f"Top 3 parameters correlated with Objective are: {top_3_params}")

fig, axes = plt.subplots(1, 3, figsize=(20, 4))
fig.suptitle('2D Scatter for Top 3 Correlated Parameters', fontsize=14)

# Plot parameter 1 vs 2, 1 vs 3, and 2 vs 3
pairs = [(top_3_params[0], top_3_params[1]), 
         (top_3_params[0], top_3_params[2]), 
         (top_3_params[1], top_3_params[2])]

for i, (px, py) in enumerate(pairs):
    sc = axes[i].scatter(df[px], df[py], c=df[target], cmap='plasma', alpha=0.8, s=30)
    axes[i].set_xlabel(px)
    axes[i].set_ylabel(py)
    axes[i].grid(True, alpha=0.3)

fig.colorbar(sc, ax=axes.ravel().tolist(), label='Objective Value')
plt.savefig(os.path.join(plot_dir, '4_top3_scatter.png'))
plt.show()

print(f"\nAnalysis complete! All plots saved to: {plot_dir}")
