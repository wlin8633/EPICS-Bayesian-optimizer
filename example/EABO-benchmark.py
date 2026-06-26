import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib')))
from BayesianOptimization import suggest_next_params_bo
from dummy_npz_generator import obj, input_bounds

def run_trial(seed, use_context=True):
    # Set seed for reproducible environmental noise for this trial
    np.random.seed(seed)
    random.seed(seed)
    
    # --- 1. Setup ---
    context_cols = ["x6", "x7"] if use_context else []
    
    if use_context:
        bo_bounds = input_bounds.copy()
    else:
        bo_bounds = {k: input_bounds[k] for k in input_bounds if k not in ["x6", "x7"]}
        
    objective_col = "obj"
    
    # --- 2. Generate Initial Data using Scrambled Sobol ---
    # Scrambled Sobol preserves uniformity but randomized positions based on the seed
    d = len(input_bounds)
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    
    # Sobol works best with powers of 2. We generate 2^5 = 32 points, and take the first 30.
    initial_samples = 30
    sobol_norm = sampler.random_base2(m=5)[:initial_samples]
    
    # Scale Sobol points from [0, 1] to actual physical bounds
    keys = list(input_bounds.keys())
    lows = np.array([input_bounds[k][0] for k in keys])
    highs = np.array([input_bounds[k][1] for k in keys])
    sobol_scaled = lows + sobol_norm * (highs - lows)
    
    data = []
    for i in range(initial_samples):
        x = sobol_scaled[i].tolist()
        y = obj(x)
        row = {k: x[idx] for idx, k in enumerate(keys)}
        row[objective_col] = y
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # --- 3. Pre-generate Fixed Environment Noise for this Trial ---
    bo_iterations = 200
    env_x6_array = np.random.uniform(input_bounds["x6"][0], input_bounds["x6"][1], bo_iterations)
    env_x7_array = np.random.uniform(input_bounds["x7"][0], input_bounds["x7"][1], bo_iterations)
    
    # --- 4. Run Optimization Loop ---
    for i in range(bo_iterations):
        env_x6 = env_x6_array[i]
        env_x7 = env_x7_array[i]
        
        if use_context:
            current_context = {"x6": env_x6, "x7": env_x7}
        else:
            current_context = None
            
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
            print(f"Error in BO at Iter {i}: {e}")
            break
            
        x_dict = {k: suggestion_list[idx] for idx, k in enumerate(bo_bounds.keys())}
        x_dict["x6"] = env_x6
        x_dict["x7"] = env_x7
        
        x_arr = [x_dict[k] for k in keys]
        y = obj(x_arr)
        
        row = x_dict.copy()
        row[objective_col] = y
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        
    # Calculate the optimization trace (cummax of objective)
    df['best_obj'] = df['obj'].cummax()
    return df['best_obj'].values, df['obj'].iloc[initial_samples:].mean()

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    N_TRIALS = 10
    
    eabo_traces = []
    trad_traces = []
    
    eabo_avg_yields = []
    trad_avg_yields = []
    
    print(f"=== Starting Benchmark with {N_TRIALS} Trials ===")
    
    for trial in range(N_TRIALS):
        seed = 42 + trial
        print(f"--- Running Trial {trial+1}/{N_TRIALS} (Seed: {seed}) ---")
        
        # Run EA-BO
        eabo_trace, eabo_yield = run_trial(seed=seed, use_context=True)
        eabo_traces.append(eabo_trace)
        eabo_avg_yields.append(eabo_yield)
        
        # Run Traditional BO
        trad_trace, trad_yield = run_trial(seed=seed, use_context=False)
        trad_traces.append(trad_trace)
        trad_avg_yields.append(trad_yield)
        
    # Convert to numpy arrays for statistics
    eabo_traces = np.array(eabo_traces)  # shape: (N_TRIALS, 230)
    trad_traces = np.array(trad_traces)
    
    # Calculate Mean and Std
    eabo_mean = eabo_traces.mean(axis=0)
    eabo_std = eabo_traces.std(axis=0)
    
    trad_mean = trad_traces.mean(axis=0)
    trad_std = trad_traces.std(axis=0)
    
    # Calculate Performance Metrics
    final_eabo_yield = np.mean(eabo_avg_yields)
    final_trad_yield = np.mean(trad_avg_yields)
    enhancement_factor = final_eabo_yield / final_trad_yield if final_trad_yield > 0 else float('inf')
    
    print("\n" + "="*50)
    print(" Benchmark Results (Averaged over 10 Trials) ")
    print("="*50)
    print(f" Traditional BO Avg Yield   : {final_trad_yield:.2f}")
    print(f" Environment-Aware BO Yield : {final_eabo_yield:.2f}")
    print(f" Mean Enhancement Factor    : {enhancement_factor:.2f}x")
    print("="*50 + "\n")
    
    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    x_axis = np.arange(len(eabo_mean))
    
    # Plot EA-BO
    plt.plot(x_axis, eabo_mean, label="Environment-Aware BO", color='blue', linewidth=2.5)
    plt.fill_between(x_axis, eabo_mean - eabo_std, eabo_mean + eabo_std, color='blue', alpha=0.2)
    
    # Plot Trad-BO
    plt.plot(x_axis, trad_mean, label="Traditional BO", color='orange', linewidth=2.5)
    plt.fill_between(x_axis, trad_mean - trad_std, trad_mean + trad_std, color='orange', alpha=0.2)
    
    plt.axvline(30, color='gray', linestyle='--', label='End of Random Init (30 shots)')
    
    plt.xlabel("Shot Number (Iteration)", fontsize=12)
    plt.ylabel("Objective Value (Current Best $f^+$)", fontsize=12)
    plt.title(f"Optimization Trace over {N_TRIALS} Random Trials", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add text annotation
    plt.text(0.5, 0.2, f"Mean Enhancement Factor: {enhancement_factor:.2f}x\n(Evaluated across {N_TRIALS} distinct environments)", 
             transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
             
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'EABO_Benchmark_Plot.png')
    plt.savefig(save_path)
    print(f"Benchmark plot saved to: {save_path}")
