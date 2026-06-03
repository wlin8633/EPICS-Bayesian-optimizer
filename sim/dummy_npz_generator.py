# %% imports
import os
import numpy as np
import AutoFireFunc as aff
import pandas as pd

# %% # ----> Prequisite settings
# spreadsheet settings
cred_path  = "credentials.json"
token_path = "token.json"
spreadsheet_url = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit"
# worksheet name
ws_name = "test"

client = aff.authorize_gspread(cred_path, token_path)
spreadsheet = client.open_by_url(spreadsheet_url)
worksheet = aff.access_worksheet(ws_name, spreadsheet)

# %%

# def y1(x):
#     return -(x[0] - 11)**2 - (x[1] - 2.5)**2 + (x[3])**2 + 20

# def y2(x):
#     return -np.sin(3 * x[2]) + np.cos(2 * x[5] * np.pi) + (x[4]/50) + (x[6]/25)

# def obj(x):
#     return y1(x) * y2(x) * (1 + 0.01*random.uniform(-1,1))  # add small noise

# %%

import numpy as np
import random

# --- CONTROL FACTORS ---
# WIDTH: Lower = sharper peak (harder to hit). Range: 0.05 (sharp) to 0.5 (broad)
WIDTH = 0.15          

# RIPPLE_FREQ: Higher = more ripples/local peaks. Range: 5 (few) to 20 (many)
RIPPLE_FREQ = 10.0    

# RIPPLE_CONTRAST: Strength of local optima. Range: 0.0 (smooth) to 0.5 (deep traps)
RIPPLE_CONTRAST = 0.2 

# --- CONFIGURATION ---
input_bounds = {
    "x1": (10, 12),
    "x2": (0, 5),
    # "x3": (0, 1),
    # "x4": (-3, 3),
    # "x5": (100, 200),
    # "x6": (0.001, 0.01),
    # "x7": (50, 150),
}

# Define the "Global Optimum" location (in physical units)
# You can change these to move the target around.
TARGET = np.array([0.4, 0.6]) # , 0.45, 0.1, 145.0, 0.005, 95.0])

def _get_norm_dist(x):
    """
    Helper: Normalizes inputs to 0-1 range and calculates 
    distance from the TARGET. 
    """
    # FIX: Force float type and flatten to ensure shape is (7,)
    # This prevents 'dtype=object' errors and handles (1, 7) inputs from BO tools
    x_arr = np.array(x, dtype=float).flatten()
    
    # Extract bounds
    lows = np.array([input_bounds[f"x{i+1}"][0] for i in range(x_arr.shape[0])])
    highs = np.array([input_bounds[f"x{i+1}"][1] for i in range(x_arr.shape[0])])
    
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

# %% create a dummy .npz file for testing
shot_number = 19
save_path = r"C:\Users\weilin\Desktop\Webber\GeminiCLI\sim_npzfolder"
is_start = "y"
while is_start.lower() in ['y', 'yes', 'true', '1']:
    shot_number += 1
    # read x from the spreadsheet for the current shot number
    df = pd.DataFrame(worksheet.get_all_records())
    x = df[df['shot-number'] == shot_number].values[0][0:2]

    dummy_data = {
        'shot_number': shot_number,
        'y1': y1(x),
        'y2': y2(x),
    }
    dummy_filename = f"dummy_{shot_number}_tp_analysis.npz"
    np.savez(os.path.join(save_path, dummy_filename), **dummy_data)
    print(f"Created dummy file: {dummy_filename}")
    is_start = input("Continue creating dummy files? (y/n): ")
# %%
