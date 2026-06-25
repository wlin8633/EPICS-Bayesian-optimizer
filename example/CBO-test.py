# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import os

# Add parent directory to path to import modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'lib'))

from lib.BayesianOptimization import suggest_next_params_bo
from dummy_npz_generator import obj, input_bounds

def run_test(use_context=True):
    print(f"--- Running Test (CBO={'ON' if use_context else 'OFF'}) ---")
    
    # 1. Setup
    context_cols = ["x6", "x7"] if use_context else []
    # The optimizer will use ALL keys in input_bounds for optimization, 
    # but we will only pass context_cols if use_context is True.
    # Wait, input_bounds has x1-x7. If we don't pass context_cols, BO optimizes x1-x7 as if they are all actions.
    # We must restrict input_bounds for the "Traditional" case to only x1-x5!
    if use_context:
        bo_bounds = input_bounds.copy()
    else:
        bo_bounds = {k: input_bounds[k] for k in input_bounds if k not in ["x6", "x7"]}

    objective_col = "obj"

    # 2. Generate Initial Random Data (15 shots)
    initial_samples = 15
    data = []
    for i in range(initial_samples):
        x = [random.uniform(input_bounds[k][0], input_bounds[k][1]) for k in input_bounds.keys()]
        y = obj(x)
        row = {k: x[idx] for idx, k in enumerate(input_bounds.keys())}
        row[objective_col] = y
        data.append(row)

    df = pd.DataFrame(data)

    # 4. Run Optimization Loop
    bo_iterations = 200
    for i in range(bo_iterations):
        # The environment ALWAYS has random context x6, x7
        env_x6 = random.uniform(input_bounds["x6"][0], input_bounds["x6"][1])
        env_x7 = random.uniform(input_bounds["x7"][0], input_bounds["x7"][1])
        
        # If CBO is ON, we pass the current context to the BO
        if use_context:
            current_context = {"x6": env_x6, "x7": env_x7}
        else:
            current_context = None
        
        # Get suggestion
        try:
            suggestion_list = suggest_next_params_bo(
                df=df,
                inputNames=bo_bounds,
                optName=objective_col,
                optType="max",
                phase="bayes",
                context_keys=context_cols,
                current_context=current_context
            )
        except Exception as e:
            print(f"Error in BO: {e}")
            break
            
        # Combine action and actual environment context to evaluate physics
        x_dict = {k: suggestion_list[idx] for idx, k in enumerate(bo_bounds.keys())}
        x_dict["x6"] = env_x6
        x_dict["x7"] = env_x7
        
        x_arr = [x_dict[k] for k in input_bounds.keys()]
        y = obj(x_arr)
        
        # Record
        row = x_dict.copy()
        row[objective_col] = y
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        
        print(f"Iter {i+1:02d} | x6: {env_x6:.4f} | Sug x1: {x_dict['x1']:5.2f} | Obj: {y:7.2f}")

    return df

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    # Run both tests
    df_cbo = run_test(use_context=True)
    df_trad = run_test(use_context=False)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Plot 1: CBO Active Tracking ---
    bo_cbo = df_cbo.iloc[15:]
    sc = axes[0, 0].scatter(bo_cbo["x6"], bo_cbo["x1"], c=bo_cbo["obj"], cmap='plasma', s=50, edgecolor='k', vmin=0, vmax=200)
    axes[0, 0].set_xlabel("Context: x6 (Prepulse)")
    axes[0, 0].set_ylabel("Action: x1 (Focus Pos)")
    axes[0, 0].set_title("Contextual BO: Action Tracking Context")
    plt.colorbar(sc, ax=axes[0, 0], label='Objective Yield')

    # Theoretical optimum line: dynamic_target[0] based on new logic
    x6_range = np.linspace(input_bounds["x6"][0], input_bounds["x6"][1], 100)
    c1_norm = (x6_range - 0.0055) / 0.0045
    drift_amp = 0.15 * 0.1 # WIDTH * 0.1
    x1_opt = 11.0 + (c1_norm * drift_amp * 2.0)
    axes[0, 0].plot(x6_range, x1_opt, 'r--', label='Theoretical Optimum', alpha=0.7)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(input_bounds["x1"])

    # --- Plot 2: Traditional BO (Blind) ---
    bo_trad = df_trad.iloc[15:]
    sc2 = axes[0, 1].scatter(bo_trad["x6"], bo_trad["x1"], c=bo_trad["obj"], cmap='plasma', s=50, edgecolor='k', vmin=0, vmax=200)
    axes[0, 1].set_xlabel("Context: x6 (Prepulse)")
    axes[0, 1].set_ylabel("Action: x1 (Focus Pos)")
    axes[0, 1].set_title("Traditional BO: Blind to Context")
    plt.colorbar(sc2, ax=axes[0, 1], label='Objective Yield')
    axes[0, 1].plot(x6_range, x1_opt, 'r--', label='Theoretical Optimum', alpha=0.7)
    axes[0, 1].legend()
    axes[0, 1].set_ylim(input_bounds["x1"])

    # --- Plot 3: Rolling Average Objective ---
    window = 5
    cbo_roll = df_cbo["obj"].rolling(window).mean()
    trad_roll = df_trad["obj"].rolling(window).mean()
    
    axes[1, 0].plot(cbo_roll, label="Contextual BO", color='blue', linewidth=2)
    axes[1, 0].plot(trad_roll, label="Traditional BO", color='orange', linewidth=2)
    axes[1, 0].axvline(15, color='gray', linestyle='--', label='End of Random Init')
    axes[1, 0].set_xlabel("Shot Number")
    axes[1, 0].set_ylabel(f"{window}-Shot Rolling Avg Objective")
    axes[1, 0].set_title("Yield Convergence Comparison")
    axes[1, 0].legend()

    # --- Plot 4: Scatter of Objectives ---
    axes[1, 1].scatter(df_cbo.index, df_cbo["obj"], label="Contextual BO", alpha=0.6, color='blue')
    axes[1, 1].scatter(df_trad.index, df_trad["obj"], label="Traditional BO", alpha=0.6, color='orange')
    axes[1, 1].axvline(15, color='gray', linestyle='--')
    axes[1, 1].set_xlabel("Shot Number")
    axes[1, 1].set_ylabel("Raw Objective Value")
    axes[1, 1].set_title("Raw Yield Stability")
    axes[1, 1].legend()

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'CBO_Comparison_Plot.png')
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    # plt.show() # Disable show for headless execution

# %%
