# %% imports
import os
# %% imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib')))
import numpy as np
import AutoFireFunc as aff
import pandas as pd
import time

# %% # ----> Prequisite settings
# spreadsheet settings
import json
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'gsheet_config.json')
try:
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    cred_path = cfg.get("cred_path", r"C:\path\to\credentials.json")
    token_path = cfg.get("token_path", r"C:\path\to\token.json")
    spreadsheet_url = cfg.get("spreadsheet_url", "https://docs.google.com/spreadsheets/d/YOUR_DOCUMENT_ID/edit")
    ws_name = cfg.get("worksheet_name", "test")
except:
    cred_path = r"C:\path\to\credentials.json"
    token_path = r"C:\path\to\token.json"
    spreadsheet_url = "https://docs.google.com/spreadsheets/d/YOUR_DOCUMENT_ID/edit"
    ws_name = "test"


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
    "x3": (0, 1),
    "x4": (-3, 3),
    "x5": (100, 200),
    "x6": (0.001, 0.01),
    "x7": (50, 150),
}

# Define the "Global Optimum" location (in physical units)
# You can change these to move the target around.
TARGET = np.array([11.0, 2.0, 0.8, 0.0, 150.0, 0.005, 100.0])

def _get_norm_dist(x):
    """
    Helper: Normalizes inputs to 0-1 range and calculates 
    distance from the TARGET. 
    """
    # FIX: Force float type and flatten to ensure shape is (7,)
    # This prevents 'dtype=object' errors and handles (1, 7) inputs from BO tools
    x_arr = np.array(x, dtype=float).flatten()
    
    # --- PHYSICS COUPLING (CROSS-CORRELATION) ---
    # Normalize context variables x6 and x7 to [-1, 1] ranges to make the drift logic general
    # x6 range is (0.001, 0.01), mid is 0.0055, half-range is 0.0045
    c1_norm = (x_arr[5] - 0.0055) / 0.0045
    # x7 range is (50, 150), mid is 100.0, half-range is 50.0
    c2_norm = (x_arr[6] - 100.0) / 50.0
    
    # Set maximum drift amplitude relative to Peak Width (e.g., 10% of WIDTH)
    drift_amplitude = WIDTH * 1  # in normalized 0-1 space

    # Shift target for x1 and x2 based on context variables
    x1_range = input_bounds["x1"][1] - input_bounds["x1"][0]
    x2_range = input_bounds["x2"][1] - input_bounds["x2"][0]
    
    dynamic_target = TARGET.copy()
    dynamic_target[0] = TARGET[0] + (c1_norm * drift_amplitude * x1_range)
    dynamic_target[1] = TARGET[1] + (c2_norm * drift_amplitude * x2_range)

    # CRITICAL: To allow EA-BO to fully recover the yield, the environment variables (x6, x7)
    # should NOT penalize the distance directly. They only act as shift parameters!
    dynamic_target[5] = x_arr[5]
    dynamic_target[6] = x_arr[6]
    # --------------------------------------------
    
    # Extract bounds
    lows = np.array([input_bounds[f"x{i+1}"][0] for i in range(x_arr.shape[0])])
    highs = np.array([input_bounds[f"x{i+1}"][1] for i in range(x_arr.shape[0])])
    
    # Normalize input and target to [0, 1]
    x_norm = (x_arr - lows) / (highs - lows)
    t_norm = (dynamic_target - lows) / (highs - lows)
    
    # Calculate vector difference
    diff = x_norm - t_norm
    
    # Euclidean distance squared
    dist_sq = np.sum(diff**2)
    
    return diff, dist_sq

def y1(x):
    """
    Simulates 'Proton Yield' (Total Charge, e.g., in pC).
    TNSA proton yield typically has a broader optimal region regarding laser focus 
    and target thickness. It is a stable, wider Gaussian envelope.
    """
    diff, dist_sq = _get_norm_dist(x)
    
    # A Gaussian that is 2x wider than the main objective (easier signal)
    val = np.exp(-dist_sq / (2 * (WIDTH * 2)**2))
    
    # Scale to typical TNSA output magnitude (e.g. 0 to 500 pC)
    # Add ~2% shot-to-shot noise
    noise = 1 + 0.02 * random.uniform(-1, 1)
    return 500.0 * val * noise

def y2(x):
    """
    Simulates 'Maximum Proton Energy' (Cutoff Energy, MeV).
    TNSA cutoff energy is highly sensitive to intensity (focus quality, prepulse, wavefront).
    It drops sharply if conditions are not perfect, and has local optima due to 
    nonlinear laser-plasma instabilities.
    """
    diff, dist_sq = _get_norm_dist(x)
    
    # 1. Main Physics Peak (Narrow Gaussian)
    envelope = np.exp(-dist_sq / (2 * WIDTH**2))
    
    # 2. Ripple Factor (Cosine modulation heavily on wavefront terms x4-x7)
    # We apply higher frequency/amplitude ripples to wavefront parameters
    ripple_weights = np.array([0.5, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    ripple_pattern = np.mean(np.cos(diff * ripple_weights * RIPPLE_FREQ * np.pi))
    
    # Combine: The ripples only exist where there is a base TNSA signal
    val = envelope * (1 + RIPPLE_CONTRAST * ripple_pattern)
    
    # Scale to typical TNSA cutoff energy magnitude (e.g. up to 25 MeV)
    # Add ~3% shot-to-shot noise for this sensitive diagnostic
    noise = 1 + 0.03 * random.uniform(-1, 1)
    return 25.0 * val * noise

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

# %% # %% GUI for Dummy Generation
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import glob

class DummyGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dummy NPZ Generator")
        
        # --- Variables ---
        self.save_path_var = tk.StringVar(value=os.path.join(os.path.dirname(__file__), "sim_npzfolder"))
        self.shot_num_var = tk.StringVar(value="1")
        self.auto_range_var = tk.StringVar(value="50")
        self.auto_prepulse_var = tk.BooleanVar(value=False)
        
        # Google Sheet Variables
        self.url_var = tk.StringVar(value=spreadsheet_url)
        self.ws_name_var = tk.StringVar(value=ws_name)
        
        self.worksheet = None
        self.is_auto_running = False
        self._setup_ui()

    def _setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- Google Sheet Settings ---
        gs_frame = ttk.LabelFrame(main_frame, text="Google Sheet Settings", padding="10")
        gs_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(gs_frame, text="Spreadsheet URL:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(gs_frame, textvariable=self.url_var, width=60).grid(row=0, column=1, columnspan=2, sticky=tk.W, pady=2)
        
        ttk.Label(gs_frame, text="Worksheet Name:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(gs_frame, textvariable=self.ws_name_var, width=20).grid(row=1, column=1, sticky=tk.W, pady=2)
        ttk.Button(gs_frame, text="Reconnect", command=self.reconnect_gsheet).grid(row=1, column=2, sticky=tk.W, padx=5)

        # --- Path Settings ---
        path_frame = ttk.LabelFrame(main_frame, text="Save Location", padding="10")
        path_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(path_frame, text="Folder Path:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(path_frame, textvariable=self.save_path_var, width=50).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(path_frame, text="Browse...", command=self.browse_folder).grid(row=0, column=2, sticky=tk.W)

        # --- Shot Number Settings ---
        shot_frame = ttk.LabelFrame(main_frame, text="Shot Settings", padding="10")
        shot_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(shot_frame, text="Next Shot Number:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(shot_frame, textvariable=self.shot_num_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(shot_frame, text="Detect Largest in Folder", command=self.detect_shot_number).grid(row=0, column=2, sticky=tk.W)

        # --- Action Buttons ---
        action_frame = ttk.LabelFrame(main_frame, text="Actions", padding="10")
        action_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(action_frame, text="Push Prepulse (Known Context)", command=self.push_prepulse).grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
        
        ttk.Button(action_frame, text="Generate Single NPZ (Manual)", command=self.generate_single).grid(row=1, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
        
        ttk.Label(action_frame, text="Auto Target Range:").grid(row=2, column=0, sticky=tk.E)
        ttk.Entry(action_frame, textvariable=self.auto_range_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        ttk.Checkbutton(action_frame, text="Auto Push Prepulse", variable=self.auto_prepulse_var).grid(row=2, column=2, sticky=tk.W, padx=5)
        
        self.auto_btn = ttk.Button(action_frame, text="Start Auto Sampling", command=self.toggle_auto)
        self.auto_btn.grid(row=2, column=3, sticky=tk.W)

        self.status_var = tk.StringVar(value="Ready. Please connect to Google Sheets.")
        ttk.Label(main_frame, textvariable=self.status_var, foreground="blue").grid(row=4, column=0, sticky=tk.W, pady=5)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.save_path_var.set(folder)

    def reconnect_gsheet(self):
        try:
            self.status_var.set("Connecting to Google Sheets...")
            self.root.update()
            client = aff.authorize_gspread(cred_path, token_path)
            spreadsheet = client.open_by_url(self.url_var.get())
            self.worksheet = aff.access_worksheet(self.ws_name_var.get(), spreadsheet)
            self.status_var.set("Connected to Google Sheets successfully!")
        except Exception as e:
            messagebox.showerror("Connection Error", str(e))
            self.status_var.set("Connection failed.")

    def detect_shot_number(self):
        folder = self.save_path_var.get()
        if not os.path.exists(folder):
            self.shot_num_var.set("1")
            return
            
        npz_files = glob.glob(os.path.join(folder, "*.npz"))
        max_num = 0
        for f in npz_files:
            basename = os.path.basename(f)
            # Assuming format: dummy_{shot_number}_tp_analysis.npz
            parts = basename.split('_')
            for p in parts:
                if p.isdigit():
                    max_num = max(max_num, int(p))
        
        self.shot_num_var.set(str(max_num + 1))
        self.status_var.set(f"Detected next shot number: {max_num + 1}")

    def push_prepulse(self, auto_shot_num=None):
        if self.worksheet is None:
            messagebox.showwarning("Warning", "Please connect to Google Sheets first.")
            return

        try:
            df = pd.DataFrame(self.worksheet.get_all_records())
            shot_col = 'shot-number' if 'shot-number' in df.columns else 'Shot'
            
            shot_num = auto_shot_num if auto_shot_num is not None else int(self.shot_num_var.get())
            
            # Context Configuration
            CONTEXT_KEYS = ["x6", "x7"]
            context_vals = {k: random.uniform(input_bounds[k][0], input_bounds[k][1]) for k in CONTEXT_KEYS}
            
            headers = self.worksheet.row_values(1)
            new_row = ["" for _ in headers]
            if shot_col in headers:
                new_row[headers.index(shot_col)] = shot_num
                
            for k, v in context_vals.items():
                if k in headers:
                    new_row[headers.index(k)] = v
                
            self.worksheet.append_row(new_row, value_input_option='USER_ENTERED')
            context_str = ", ".join([f"{k}={v:.5f}" for k, v in context_vals.items()])
            self.status_var.set(f"Pushed Context for Shot {shot_num}: {context_str}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _generate_npz_for_shot(self, shot_num, fail_silently=False, df=None):
        if self.worksheet is None:
            raise ValueError("Not connected to Google Sheets.")

        if df is None:
            df = pd.DataFrame(self.worksheet.get_all_records())
        shot_col = 'shot-number' if 'shot-number' in df.columns else 'Shot'
        
        # Check if shot exists
        row_data = df[df[shot_col] == shot_num]
        if row_data.empty:
            if fail_silently:
                return None
            raise ValueError(f"Shot {shot_num} not found in spreadsheet! Run Optimizer first.")
            
        # Build x safely, handling missing columns if the user is running traditional BO
        x_vals = []
        for col in input_bounds.keys():
            if col in df.columns:
                val = row_data[col].values[0]
                x_vals.append(val)
            else:
                x_vals.append("")
        x = np.array(x_vals, dtype=object)
        
        # For Active Mode (prepulse pushed), actions might be missing. We shouldn't generate NPZ if actions are missing!
        action_cols = [k for k in input_bounds.keys() if k not in ["x6", "x7"]]
        for idx, col in enumerate(input_bounds.keys()):
            if col in action_cols and (pd.isna(x[idx]) or x[idx] == ""):
                if fail_silently:
                    return None
                raise ValueError(f"Action {col} is missing for Shot {shot_num}! Waiting for BO.")
        
        # If any context/input is empty, generate it randomly
        for idx, (col_name, bound) in enumerate(input_bounds.items()):
            if pd.isna(x[idx]) or x[idx] == "":
                x[idx] = random.uniform(bound[0], bound[1])

        dummy_data = {
            'shot_number': shot_num,
            'y1': y1(x),
            'y2': y2(x),
        }
        for idx, col_name in enumerate(input_bounds.keys()):
            dummy_data[col_name] = x[idx]
            
        save_path = self.save_path_var.get()
        os.makedirs(save_path, exist_ok=True)
        dummy_filename = f"dummy_{shot_num}_tp_analysis.npz"
        np.savez(os.path.join(save_path, dummy_filename), **dummy_data)
        return dummy_filename

    def generate_single(self):
        try:
            shot_num = int(self.shot_num_var.get())
            filename = self._generate_npz_for_shot(shot_num)
            self.status_var.set(f"Generated: {filename}")
            self.shot_num_var.set(str(shot_num + 1))
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Generation failed.")

    def toggle_auto(self):
        if self.worksheet is None:
            messagebox.showwarning("Warning", "Please connect to Google Sheets first.")
            return

        if self.is_auto_running:
            self.is_auto_running = False
            self.auto_btn.config(text="Start Auto Sampling")
            self.status_var.set("Auto sampling stopped.")
        else:
            try:
                target_count = int(self.auto_range_var.get())
                self.is_auto_running = True
                self.auto_btn.config(text="Stop Auto Sampling")
                threading.Thread(target=self._auto_loop, args=(target_count,), daemon=True).start()
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid integer for auto range.")

    def _auto_loop(self, target_count):
        count = 0
        while self.is_auto_running and count < target_count:
            try:
                shot_num = int(self.shot_num_var.get())
                
                # Check current sheet status
                df = pd.DataFrame(self.worksheet.get_all_records())
                shot_col = 'shot-number' if 'shot-number' in df.columns else 'Shot'
                row_data = df[df[shot_col] == shot_num]
                
                # If row doesn't exist and we need to push prepulse, do it
                if self.auto_prepulse_var.get() and row_data.empty:
                    self.push_prepulse(shot_num)
                    time.sleep(2) # Give GS a moment
                    continue
                
                # Attempt to generate NPZ (fails silently if actions or row are missing)
                # Pass the df we just downloaded so we don't hit the API again
                filename = self._generate_npz_for_shot(shot_num, fail_silently=True, df=df)
                
                if filename is None:
                    # Still waiting for BO to populate the row
                    self.root.after(0, lambda s=shot_num: self.status_var.set(f"Waiting for BO Action for Shot {s}..."))
                    time.sleep(5) # Increased to 5s to avoid hitting 60 reads/min API quota
                    continue
                
                # Successfully generated
                self.root.after(0, lambda f=filename, s=shot_num: self.status_var.set(f"Auto Generated: {f}"))
                self.root.after(0, lambda s=shot_num: self.shot_num_var.set(str(s + 1)))
                
                count += 1
                time.sleep(15)  # Simulate 15s interval between shots
            except Exception as e:
                self.root.after(0, lambda err=e: self.status_var.set(f"Auto loop paused: {err}"))
                time.sleep(3) # Don't break loop, just retry
                
        if count >= target_count:
            self.is_auto_running = False
            self.root.after(0, lambda: self.auto_btn.config(text="Start Auto Sampling"))
            self.root.after(0, lambda: self.status_var.set(f"Auto sampling completed ({target_count} shots)."))

if __name__ == "__main__":
    root = tk.Tk()
    app = DummyGeneratorGUI(root)
    root.mainloop()
