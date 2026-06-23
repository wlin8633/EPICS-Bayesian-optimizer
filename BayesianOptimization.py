# %% Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skopt import Optimizer
from skopt.space import Real
import random

# Simulation of an optimization process using Bayesian Optimization with local search

def _clean_history(df, inputNames, optName):
    keys = list(inputNames.keys())
    cols_needed = keys + [optName]

    df_cleaned = df.dropna(subset=cols_needed).copy()
    for col in cols_needed:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")
    df_cleaned = df_cleaned.dropna(subset=cols_needed)

    return df_cleaned


def _objective_to_skopt_y(vals, optType):
    # skopt minimizes; if user wants max, minimize -objective
    vals = list(vals)
    return vals if optType == "min" else [-v for v in vals]


def _sample_uniform(inputNames):
    # returns list [x1, x2, ...] in the same key order
    keys = list(inputNames.keys())
    return [random.uniform(*inputNames[k]) for k in keys]


def _local_perturb(best_x, inputNames, frac=0.10):
    """
    ±frac around best for each dimension, clipped to global bounds.
    If a bound includes 0 or very small range, still behaves.
    """
    keys = list(inputNames.keys())
    x_new = []
    for i, k in enumerate(keys):
        lo, hi = inputNames[k]
        span = hi - lo
        # perturbation scale: ± frac of full range (more stable than ±frac of value)
        delta = frac * span
        v = best_x[i] + random.uniform(-delta, delta)
        v = min(hi, max(lo, v))
        x_new.append(v)
    return x_new


def suggest_next_params_bo(
    df,
    inputNames,
    optName,
    optType="min",
    phase="bayes",
    random_state=None,
    local_frac=0.10,
):
    """
    Suggest ONE next point given current df.
    phase:
      - "random": ignore GP, just uniform random within bounds
      - "bayes": fit GP to df and ask EI point
      - "local": local perturb around current best (±10% of range)
    """
    keys = list(inputNames.keys())

    df_cleaned = _clean_history(df, inputNames, optName)

    if phase == "random":
        return _sample_uniform(inputNames)

    if df_cleaned.empty:
        # If no usable history, fall back to random
        return _sample_uniform(inputNames)

    # find best_x for local scan / also useful generally
    if optType == "min":
        best_idx = df_cleaned[optName].idxmin()
    else:
        best_idx = df_cleaned[optName].idxmax()
    best_x = df_cleaned.loc[best_idx, keys].to_numpy().tolist()

    if phase == "local":
        return _local_perturb(best_x, inputNames, frac=local_frac)

    # phase == "bayes"
    history_x_arr = df_cleaned[keys].to_numpy(dtype=float)
    
    # Clip history_x to exactly within the bounds to prevent skopt errors
    # (e.g., due to floating point precision or manually entered spreadsheet data)
    for i, k in enumerate(keys):
        lo, hi = inputNames[k]
        history_x_arr[:, i] = np.clip(history_x_arr[:, i], lo, hi)
        
    history_x = history_x_arr.tolist()
    history_obj = df_cleaned[optName].to_numpy().tolist()
    history_y = _objective_to_skopt_y(history_obj, optType)

    dimensions = [Real(lo, hi, name=name) for name, (lo, hi) in inputNames.items()]
    optimizer = Optimizer(
        dimensions=dimensions,
        base_estimator="GP",
        acq_func="EI",
        acq_optimizer="auto",
        random_state=random_state,
    )

    # preload known data
    optimizer.tell(history_x, history_y)

    # get next EI suggestion
    return optimizer.ask()

# -----------------------------
# Example: 3-phase optimization loop
# -----------------------------
if __name__ == "__main__":
    
    optName = "obj"
    optType = "max"
    local_frac = 0.01  # ±10% local scan

    # --- CONTROL FACTORS ---
    # WIDTH: Lower = sharper peak (harder to hit). Range: 0.05 (sharp) to 0.5 (broad)
    WIDTH = 0.15          

    # RIPPLE_FREQ: Higher = more ripples/local peaks. Range: 5 (few) to 20 (many)
    RIPPLE_FREQ = 10.0    

    # RIPPLE_CONTRAST: Strength of local optima. Range: 0.0 (smooth) to 0.5 (deep traps)
    RIPPLE_CONTRAST = 0.2 

    # --- CONFIGURATION ---
    inputNames = {
        "x1": (10, 12),
        "x2": (0, 5),
        "x3": (0, 1),
        "x4": (-3, 3),
        "x5": (100, 200),
        "x6": (0.001, 0.01),
        "x7": (50, 150),
    }

    # Define the "Global Optimum" location (in physical units)
    # You can change these to move the target around.
    TARGET = np.array([11.2, 2.5, 0.45, 0.1, 145.0, 0.005, 95.0])

    def _get_norm_dist(x):
        """
        Helper: Normalizes inputs to 0-1 range and calculates 
        distance from the TARGET. 
        """
        # FIX: Force float type and flatten to ensure shape is (7,)
        # This prevents 'dtype=object' errors and handles (1, 7) inputs from BO tools
        x_arr = np.array(x, dtype=float).flatten()
        
        # Extract bounds
        lows = np.array([inputNames[f"x{i+1}"][0] for i in range(7)])
        highs = np.array([inputNames[f"x{i+1}"][1] for i in range(7)])
        
        # Normalize input and target to [0, 1]
        x_norm = (x_arr - lows) / (highs - lows)
        t_norm = (TARGET - lows) / (highs - lows)
        
        # Calculate vector difference
        diff = x_norm - t_norm
        
        # Euclidean distance squared
        dist_sq = np.sum(diff**2)
        
        return diff, dist_sq

    def y1(x):
        """
        Simulates 'Total Charge' or 'Energy'.
        Behavior: A broader, stable Gaussian envelope. Easier to find.
        """
        _, dist_sq = _get_norm_dist(x)
        
        # A Gaussian that is 2x wider than the main objective (easier signal)
        val = np.exp(-dist_sq / (2 * (WIDTH * 2)**2))
        
        # Scale to typical PC output magnitude (e.g. 0 to 100 pC)
        return 100 * val

    def y2(x):
        """
        Simulates 'Beam Quality' or 'Flux'.
        Behavior: Highly sensitive, contains the ripples/local optima.
        """
        diff, dist_sq = _get_norm_dist(x)
        
        # 1. Main Physics Peak (Narrow Gaussian)
        envelope = np.exp(-dist_sq / (2 * WIDTH**2))
        
        # 2. Ripple Factor (Cosine modulation on all axes)
        # Creates a lattice of local maxima nearby
        # We take the average cosine to ensure it doesn't kill the signal completely
        ripple_pattern = np.mean(np.cos(diff * RIPPLE_FREQ * np.pi))
        
        # Combine: The ripples only exist where there is signal (inside the envelope)
        val = envelope * (1 + RIPPLE_CONTRAST * ripple_pattern)
        
        # Scale to typical flux magnitude
        return 50 * val

    def obj(x):
        """
        The Optimization Objective.
        Product of y1 and y2 + Noise.
        """
        val_y1 = y1(x)
        val_y2 = y2(x)
        
        # Physics relationship: usually Charge * Quality
        pure_obj = val_y1 * val_y2
        
        # Add 2% relative noise (shot-to-shot fluctuation)
        noise = 1 + 0.02 * random.uniform(-1, 1)
        
        return pure_obj * noise

    # Seed with *optional* historical data (can be empty)
    df_sim = pd.DataFrame(columns=list(inputNames.keys()) + [y1.__name__, y2.__name__, optName])

    # ---- Phase 1: Random exploration (10–15 points)
    N_random = 5 * len(inputNames)
    for i in range(N_random):
        x = suggest_next_params_bo(df_sim, inputNames, optName, optType, phase="random")
        df_sim.loc[len(df_sim)] = x + [y1(x), y2(x), obj(x)]
        print(f"[Random {i+1:02d}] " + ", ".join(f"{k}={v:.4f}" for k, v in zip(inputNames.keys(), x))
            + f", obj={df_sim.iloc[-1][optName]:.4f}")

    # ---- Phase 2: Bayesian optimization (15–30 points)
    N_bayes = 20 * len(inputNames)
    for i in range(N_bayes):
        x = suggest_next_params_bo(df_sim, inputNames, optName, optType, phase="bayes", random_state=None)
        df_sim.loc[len(df_sim)] = x + [y1(x), y2(x), obj(x)]
        print(f"[BO     {i+1:02d}] " + ", ".join(f"{k}={v:.4f}" for k, v in zip(inputNames.keys(), x))
            + f", obj={df_sim.iloc[-1][optName]:.4f}")

    # ---- Phase 3: Local scan around best (±10% of range)
    N_local = 5 * len(inputNames)
    for i in range(N_local):
        x = suggest_next_params_bo(df_sim, inputNames, optName, optType, phase="local", local_frac=local_frac)
        df_sim.loc[len(df_sim)] = x + [y1(x), y2(x), obj(x)]
        print(f"[Local  {i+1:02d}] " + ", ".join(f"{k}={v:.4f}" for k, v in zip(inputNames.keys(), x))
            + f", obj={df_sim.iloc[-1][optName]:.4f}")

    # Report best
    if optType == "max":
        best_row = df_sim.loc[df_sim[optName].idxmax()]
    else:
        best_row = df_sim.loc[df_sim[optName].idxmin()]

    print("\nBest found:")
    print(best_row)

    # Plot optimization progress
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(df_sim[optName].values, marker='o')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(optName)
    ax.set_title('Optimization Progress')
    plt.show()
# %%
