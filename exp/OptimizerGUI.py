import tkinter as tk
from tkinter import messagebox, filedialog
from ophyd.signal import EpicsSignal
import pandas as pd
import gspread
import numpy as np
import ast
import json
import os
import threading
import time
import glob
import importlib.util

# Assuming these files are in the same directory
import AutoFireFunc as aff
import BayesianOptimization as bo
from ImageMonitor import MonitorWindow
from TPAnalyzer import TPAnalyzerWindow

import pathlib


class OptimizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bayesian Optimization Assistant")
        self.script_dir = pathlib.Path(__file__).parent
        self.client = None  # For lazy initialization of gspread client
        self.state_file = "gui_state.json"
        self.last_suggestion = None  # To store the latest suggestion
        self.monitor_window = None # To hold the monitor window instance
        self.tp_analyzer_window = None # To hold the TP analyzer window instance
        self.settings_window = None

        # --- UI Elements ---
        frame = tk.Frame(root, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # --- Variable Declarations ---
        self.cred_path_var = tk.StringVar()
        self.token_path_var = tk.StringVar()
        self.url_var = tk.StringVar()
        self.sheet_var = tk.StringVar()
        self.shot_num_col_var = tk.StringVar()
        self.inputs_var = tk.StringVar()
        self.bounds_var = tk.StringVar()
        self.outputs_var = tk.StringVar()
        self.obj_name_var = tk.StringVar()
        self.obj_type_var = tk.StringVar()
        self.obj_formula_var = tk.StringVar()
        self.phase_var = tk.StringVar()
        self.local_frac_var = tk.StringVar()
        self.suggestion_var = tk.StringVar(value="Suggestion will appear here.")
        self.best_var = tk.StringVar(value="Current best will appear here.")

        # Auto-upload variables
        self.analysis_folder_var = tk.StringVar()
        self.output_extraction_func_var = tk.StringVar()
        self.prerequisite_func_var = tk.StringVar()
        self.file_pattern_var = tk.StringVar()
        self.autoupload_status_var = tk.StringVar(value="Status: Idle")
        self.autoupload_thread = None
        self.stop_autoupload_event = threading.Event()
        self.auto_optimize_var = tk.BooleanVar()
        self.append_to_sheet_var = tk.BooleanVar(value=True)
        self.update_epics_var = tk.BooleanVar(value=True)
        self.dynamic_bg_subtraction_var = tk.BooleanVar(value=False)

        # --- UI Layout ---
        # GSheet Settings Frame
        gsheet_frame = tk.LabelFrame(frame, text="Google Sheet Settings", padx=10, pady=10)
        gsheet_frame.pack(fill=tk.X, expand=True, pady=5)

        tk.Label(gsheet_frame, text="Credentials Path:").grid(row=0, column=0, sticky=tk.W, pady=2)
        tk.Entry(gsheet_frame, textvariable=self.cred_path_var, width=80).grid(row=0, column=1, columnspan=3, sticky=tk.W)
        tk.Label(gsheet_frame, text="Token Path:").grid(row=1, column=0, sticky=tk.W, pady=2)
        tk.Entry(gsheet_frame, textvariable=self.token_path_var, width=80).grid(row=1, column=1, columnspan=3, sticky=tk.W)
        tk.Label(gsheet_frame, text="GSpread URL:").grid(row=2, column=0, sticky=tk.W, pady=2)
        tk.Entry(gsheet_frame, textvariable=self.url_var, width=80).grid(row=2, column=1, columnspan=3, sticky=tk.W)
        tk.Label(gsheet_frame, text="Sheet Name:").grid(row=3, column=0, sticky=tk.W, pady=2)
        tk.Entry(gsheet_frame, textvariable=self.sheet_var, width=30).grid(row=3, column=1, sticky=tk.W)
        tk.Label(gsheet_frame, text="Shot Number Column:").grid(row=4, column=0, sticky=tk.W, pady=2)
        tk.Entry(gsheet_frame, textvariable=self.shot_num_col_var, width=30).grid(row=4, column=1, sticky=tk.W)

        # Optimizer Settings Frame
        optimizer_frame = tk.LabelFrame(frame, text="Optimizer Settings", padx=10, pady=10)
        optimizer_frame.pack(fill=tk.X, expand=True, pady=5)

        tk.Label(optimizer_frame, text="Input Cols (comma-sep):").grid(row=0, column=0, sticky=tk.W, pady=2)
        tk.Entry(optimizer_frame, textvariable=self.inputs_var, width=80).grid(row=0, column=1, columnspan=3, sticky=tk.W)
        tk.Label(optimizer_frame, text="Input Bounds (tuples):").grid(row=1, column=0, sticky=tk.W, pady=2)
        tk.Entry(optimizer_frame, textvariable=self.bounds_var, width=80).grid(row=1, column=1, columnspan=3, sticky=tk.W)
        tk.Label(optimizer_frame, text="Output Cols (comma-sep):").grid(row=2, column=0, sticky=tk.W, pady=2)
        tk.Entry(optimizer_frame, textvariable=self.outputs_var, width=80).grid(row=2, column=1, columnspan=3, sticky=tk.W)
        tk.Label(optimizer_frame, text="Objective:").grid(row=3, column=0, sticky=tk.W, pady=5)
        tk.Entry(optimizer_frame, textvariable=self.obj_name_var, width=15).grid(row=3, column=1, sticky=tk.W)
        tk.Entry(optimizer_frame, textvariable=self.obj_type_var, width=5).grid(row=3, column=2, sticky=tk.W)
        tk.Entry(optimizer_frame, textvariable=self.obj_formula_var, width=30).grid(row=3, column=3, sticky=tk.W)
        tk.Label(optimizer_frame, text="Phase:").grid(row=4, column=0, sticky=tk.W, pady=5)
        phase_frame = tk.Frame(optimizer_frame)
        tk.Radiobutton(phase_frame, text="Random", variable=self.phase_var, value="random").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(phase_frame, text="Bayes", variable=self.phase_var, value="bayes").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(phase_frame, text="Local", variable=self.phase_var, value="local").pack(side=tk.LEFT, padx=5)
        tk.Label(phase_frame, text="Local Frac:").pack(side=tk.LEFT, padx=(10, 2))
        tk.Entry(phase_frame, textvariable=self.local_frac_var, width=8).pack(side=tk.LEFT)
        phase_frame.grid(row=4, column=1, columnspan=3, sticky=tk.W)

        # --- Auto-Uploader Frame ---
        autoupload_frame = tk.LabelFrame(frame, text="Auto-Upload Analysis Results", padx=10, pady=10)
        autoupload_frame.pack(fill=tk.X, expand=True, pady=5)

        tk.Label(autoupload_frame, text="Analysis Folder to Monitor:").grid(row=0, column=0, sticky=tk.W, pady=2)
        tk.Entry(autoupload_frame, textvariable=self.analysis_folder_var, width=60).grid(row=0, column=1, sticky=tk.W)
        tk.Button(autoupload_frame, text="Browse...", command=self._browse_analysis_folder).grid(row=0, column=2, sticky=tk.W, padx=5)
        
        tk.Label(autoupload_frame, text="Output Extraction Func:").grid(row=1, column=0, sticky=tk.W, pady=2)
        tk.Entry(autoupload_frame, textvariable=self.output_extraction_func_var, width=40).grid(row=1, column=1, sticky=tk.W)
        tk.Label(autoupload_frame, text="File Pattern (glob):").grid(row=1, column=2, sticky=tk.E, pady=2)
        tk.Entry(autoupload_frame, textvariable=self.file_pattern_var, width=20).grid(row=1, column=3, sticky=tk.W)
        
        tk.Label(autoupload_frame, text="Prerequisite Func:").grid(row=2, column=0, sticky=tk.W, pady=2)
        tk.Entry(autoupload_frame, textvariable=self.prerequisite_func_var, width=40).grid(row=2, column=1, sticky=tk.W)

        autoupload_button_frame = tk.Frame(autoupload_frame)
        self.start_autoupload_button = tk.Button(autoupload_button_frame, text="Start Auto-Upload", command=self.start_autoupload)
        self.start_autoupload_button.pack(side=tk.LEFT, padx=5)
        self.stop_autoupload_button = tk.Button(autoupload_button_frame, text="Stop Auto-Upload", command=self.stop_autoupload, state=tk.DISABLED)
        self.stop_autoupload_button.pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(autoupload_button_frame, text="Auto-Optimization Loop", variable=self.auto_optimize_var).pack(side=tk.LEFT, padx=10)
        autoupload_button_frame.grid(row=3, column=0, columnspan=4, pady=5)
        
        tk.Label(autoupload_frame, textvariable=self.autoupload_status_var).grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=2)


        # --- Action Buttons ---
        button_frame = tk.Frame(frame)
        button_frame.pack(fill=tk.X, expand=True, pady=5)
        self.run_button = tk.Button(button_frame, text="Update and Suggest", command=self.run_optimization)
        self.run_button.pack(side=tk.LEFT, padx=5)
        self.append_button = tk.Button(button_frame, text="Append Suggestion to Sheet", command=self.append_to_sheet)
        self.append_button.pack(side=tk.LEFT, padx=5)
        self.monitor_button = tk.Button(button_frame, text="Open Image Monitor", command=self.open_monitor_window)
        self.monitor_button.pack(side=tk.LEFT, padx=5)
        self.tp_analyzer_button = tk.Button(button_frame, text="TP Analyzer", command=self.open_tp_analyzer_window)
        self.tp_analyzer_button.pack(side=tk.LEFT, padx=5)
        self.settings_button = tk.Button(button_frame, text="Settings", command=self.open_settings_window)
        self.settings_button.pack(side=tk.LEFT, padx=5)

        # --- Results Display ---
        results_frame = tk.LabelFrame(frame, text="Optimization Suggestion", padx=10, pady=10)
        results_frame.pack(fill=tk.X, expand=True, pady=5)
        tk.Label(results_frame, textvariable=self.suggestion_var, font=("Courier", 10), justify=tk.LEFT).pack(anchor=tk.W, pady=2)
        tk.Label(results_frame, textvariable=self.best_var, font=("Courier", 10), justify=tk.LEFT).pack(anchor=tk.W, pady=2)

        # --- Load previous state and start auto-save ---
        self.load_state()
        self.periodic_save()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def open_settings_window(self):
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return

        self.settings_window = tk.Toplevel(self.root)
        self.settings_window.title("Settings")
        
        settings_frame = tk.Frame(self.settings_window, padx=10, pady=10)
        settings_frame.pack(fill=tk.BOTH, expand=True)

        tk.Checkbutton(settings_frame, text="Append to Google Sheet", variable=self.append_to_sheet_var).pack(anchor=tk.W, pady=5)
        tk.Checkbutton(settings_frame, text="Update EPICS PVs", variable=self.update_epics_var).pack(anchor=tk.W, pady=5)
        tk.Checkbutton(settings_frame, text="Dynamic Background Subtraction", variable=self.dynamic_bg_subtraction_var).pack(anchor=tk.W, pady=5)

    def open_monitor_window(self):
        if self.monitor_window is None or not self.monitor_window.is_alive():
            self.monitor_window = MonitorWindow(self.root, dynamic_bg_subtraction_var=self.dynamic_bg_subtraction_var)
        else:
            self.monitor_window.lift()

    def open_tp_analyzer_window(self):
        # Ensure the monitor window exists before opening the analyzer
        self.open_monitor_window()
        
        if self.tp_analyzer_window is None or not self.tp_analyzer_window.is_alive():
            # Pass the monitor instance to the analyzer window
            self.tp_analyzer_window = TPAnalyzerWindow(self.root, monitor_window=self.monitor_window)
        else:
            self.tp_analyzer_window.lift()

    def load_state(self):
        default_state = {
            "cred_path": "credentials.json",
            "token_path": "token.json",
            "url": "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit",
            "sheet": "test",
            "shot_num_col": "Shot",
            "inputs": '"x1", "x2", "x3", "x4", "x5", "x6", "x7"',
            "bounds": '(10, 12), (0, 5), (0, 1), (-3, 3), (100, 200), (0.001, 0.01), (50, 150)',
            "outputs": 'y1, y2',
            "obj_name": "obj",
            "obj_type": "max",
            "obj_formula": "y[0] * y[1]",
            "phase": "bayes",
            "local_frac": "0.1",
            "analysis_folder": "",
            "output_extraction_func_file": "sim_user_defined_function.py",
            "prerequisite_func_file": "",
            "file_pattern": "*_tp_analysis.npz"
        }
        try:
            state_path = self._resolve_path(self.state_file)
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
            else:
                state = default_state
        except (IOError, json.JSONDecodeError):
            state = default_state

        self.cred_path_var.set(state.get("cred_path", default_state["cred_path"]))
        self.token_path_var.set(state.get("token_path", default_state["token_path"]))
        self.url_var.set(state.get("url", default_state["url"]))
        self.sheet_var.set(state.get("sheet", default_state["sheet"]))
        self.shot_num_col_var.set(state.get("shot_num_col", default_state["shot_num_col"]))
        self.inputs_var.set(state.get("inputs", default_state["inputs"]))
        self.bounds_var.set(state.get("bounds", default_state["bounds"]))
        self.outputs_var.set(state.get("outputs", default_state["outputs"]))
        self.obj_name_var.set(state.get("obj_name", default_state["obj_name"]))
        self.obj_type_var.set(state.get("obj_type", default_state["obj_type"]))
        self.obj_formula_var.set(state.get("obj_formula", default_state["obj_formula"]))
        self.phase_var.set(state.get("phase", default_state["phase"]))
        self.local_frac_var.set(state.get("local_frac", default_state["local_frac"]))
        self.analysis_folder_var.set(state.get("analysis_folder", default_state["analysis_folder"]))
        self.output_extraction_func_var.set(state.get("output_extraction_func_file", default_state["output_extraction_func_file"]))
        self.prerequisite_func_var.set(state.get("prerequisite_func_file", default_state["prerequisite_func_file"]))
        self.file_pattern_var.set(state.get("file_pattern", default_state["file_pattern"]))
        self.auto_optimize_var.set(state.get("auto_optimize", False))
        self.append_to_sheet_var.set(state.get("append_to_sheet", True))
        self.update_epics_var.set(state.get("update_epics", True))
        self.dynamic_bg_subtraction_var.set(state.get("dynamic_bg_subtraction", False))

    def save_state(self):
        state = {
            "cred_path": self.cred_path_var.get(),
            "token_path": self.token_path_var.get(),
            "url": self.url_var.get(),
            "sheet": self.sheet_var.get(),
            "shot_num_col": self.shot_num_col_var.get(),
            "inputs": self.inputs_var.get(),
            "bounds": self.bounds_var.get(),
            "outputs": self.outputs_var.get(),
            "obj_name": self.obj_name_var.get(),
            "obj_type": self.obj_type_var.get(),
            "obj_formula": self.obj_formula_var.get(),
            "phase": self.phase_var.get(),
            "local_frac": self.local_frac_var.get(),
            "analysis_folder": self.analysis_folder_var.get(),
            "output_extraction_func_file": self.output_extraction_func_var.get(),
            "prerequisite_func_file": self.prerequisite_func_var.get(),
            "file_pattern": self.file_pattern_var.get(),
            "auto_optimize": self.auto_optimize_var.get(),
            "append_to_sheet": self.append_to_sheet_var.get(),
            "update_epics": self.update_epics_var.get(),
            "dynamic_bg_subtraction": self.dynamic_bg_subtraction_var.get()
        }
        try:
            state_path = self._resolve_path(self.state_file)
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=4)
        except IOError as e:
            print(f"Error saving state: {e}")

    def periodic_save(self):
        self.save_state()
        self.root.after(30000, self.periodic_save)

    def _resolve_path(self, path_str):
        if not path_str or os.path.isabs(path_str):
            return path_str
        return str(self.script_dir / path_str)

    def _execute_user_function(self, file_path, function_name, auto_mode=False):
        resolved_path = self._resolve_path(file_path)
        if not resolved_path or not os.path.isfile(resolved_path):
            # Silently ignore if no file is provided or found.
            if file_path: # Only print warning if a path was actually provided
                print(f"Warning: User function file not found: {resolved_path}")
            return True # Indicate success to not block the process

        try:
            # Create a unique module name to avoid conflicts if loading multiple times
            module_name = f"user_module_{function_name}_{os.path.basename(resolved_path)}"
            spec = importlib.util.spec_from_file_location(module_name, resolved_path)
            user_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(user_module)
            
            if hasattr(user_module, function_name):
                func = getattr(user_module, function_name)
                func() # Execute the function
                print(f"Successfully executed user function '{function_name}' from {resolved_path}")
                return True
            else:
                raise AttributeError(f"Function '{function_name}' not found in {resolved_path}")

        except Exception as e:
            error_msg = f"Error executing user function from {resolved_path}: {e}"
            if not auto_mode:
                messagebox.showerror("User Function Error", error_msg)
            else:
                print(f"Auto-Optimization ERROR: {error_msg}")
                self.autoupload_status_var.set(f"Status: User Func Error - {e}")
            return False # Indicate failure

    def on_closing(self):
        self.save_state()
        if self.monitor_window and self.monitor_window.is_alive():
            self.monitor_window.on_closing()
        if self.tp_analyzer_window and self.tp_analyzer_window.is_alive():
            self.tp_analyzer_window.on_closing()
        self.root.destroy()

    def _run_and_append_safely(self):
        """
        Safely runs the optimization and appends the result from the main GUI thread.
        This is intended to be called via `root.after()` from a background thread.
        """
        print("Auto-Optimization: Running optimization...")
        self.run_optimization()
        
        if self.last_suggestion:
            print("Auto-Optimization: Appending suggestion to sheet and updating EPICS...")
            # auto=True bypasses the user confirmation dialog for EPICS updates.
            self.append_to_sheet(auto=True)
        else:
            print("Auto-Optimization: No new suggestion was generated, skipping append.")

    def start_autoupload(self):
        if self.autoupload_thread and self.autoupload_thread.is_alive():
            messagebox.showinfo("Info", "Auto-upload is already running.")
            return

        self.stop_autoupload_event.clear()
        self.autoupload_thread = threading.Thread(target=self._monitor_analysis_folder, daemon=True)
        self.autoupload_thread.start()

        self.start_autoupload_button.config(state=tk.DISABLED)
        self.stop_autoupload_button.config(state=tk.NORMAL)
        self.autoupload_status_var.set("Status: Monitoring...")

    def stop_autoupload(self):
        self.stop_autoupload_event.set()
        self.start_autoupload_button.config(state=tk.NORMAL)
        self.stop_autoupload_button.config(state=tk.DISABLED)
        # self.autoupload_status_var.set("Status: Stopped by user.")

    def _monitor_analysis_folder(self):
        folder_path = self._resolve_path(self.analysis_folder_var.get())
        file_pattern = self.file_pattern_var.get()
        user_func_file = self._resolve_path(self.output_extraction_func_var.get())
        shot_col_name = self.shot_num_col_var.get()

        if not os.path.isdir(folder_path):
            self.autoupload_status_var.set("Status: Error - Invalid folder path.")
            self.stop_autoupload()
            return
        if not os.path.isfile(user_func_file):
            self.autoupload_status_var.set(f"Status: Error - Function file '{user_func_file}' not found.")
            self.stop_autoupload()
            return
        if not shot_col_name:
            self.autoupload_status_var.set("Status: Error - Shot Number Column not set.")
            self.stop_autoupload()
            return

        # --- Dynamically load the user-defined function ---
        try:
            spec = importlib.util.spec_from_file_location("user_module", user_func_file)
            user_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(user_module)
            process_func = user_module.process_analysis_file
        except Exception as e:
            self.autoupload_status_var.set(f"Status: Error loading function file: {e}")
            self.stop_autoupload()
            return

        seen_files = set(glob.glob(os.path.join(folder_path, file_pattern)))
        
        worksheet = None # Lazy load the worksheet

        while not self.stop_autoupload_event.is_set():
            try:
                time.sleep(2) # Check every 2 seconds
                current_files = set(glob.glob(os.path.join(folder_path, file_pattern)))
                new_files = sorted(list(current_files - seen_files), key=os.path.getmtime)

                if new_files:
                    if worksheet is None:
                        self.autoupload_status_var.set("Status: Connecting to Google Sheet...")
                        worksheet = self._get_worksheet()
                        if worksheet is None:
                            self.autoupload_status_var.set("Status: Error - Could not connect to GSheet.")
                            self.stop_autoupload()
                            return

                    for file_path in new_files:
                        self.autoupload_status_var.set(f"Status: Processing {os.path.basename(file_path)}...")
                        
                        # Add a delay and retry to handle file writing locks
                        results = None
                        for _ in range(3):
                            try:
                                results = process_func(file_path)
                                break
                            except Exception as e:
                                print(f"Attempt to process {file_path} failed, retrying... Error: {e}")
                                time.sleep(0.5)
                        
                        if results and 'shot_number' in results:
                            shot_num = results.pop('shot_number')
                            aff.update_sheet_with_analysis(worksheet, shot_col_name, shot_num, results)
                            self.autoupload_status_var.set(f"Status: Uploaded results for shot {shot_num}.")

                            # --- NEW: Trigger auto-optimization ---
                            if self.auto_optimize_var.get():
                                self.autoupload_status_var.set(f"Status: Shot {shot_num} uploaded. Triggering next optimization cycle.")
                                self.root.after(0, self._run_and_append_safely)
                        else:
                            self.autoupload_status_var.set(f"Status: Warning - Processing failed for {os.path.basename(file_path)} either no results or missing 'shot_number'.")

                seen_files = current_files

            except Exception as e:
                self.autoupload_status_var.set(f"Status: Error in monitoring loop: {e}")
                time.sleep(5) # Wait longer after an error
        
        self.autoupload_status_var.set("Status: Idle")

    def _parse_str_list(self, s):
        return [item.strip().strip('"\'') for item in s.split(',')]

    def _parse_bounds(self, bounds_str):
        if not bounds_str.strip():
            return []
        try:
            bounds_list = ast.literal_eval(f"[{bounds_str}]")
            if not isinstance(bounds_list, list) or not all(isinstance(t, tuple) and len(t) == 2 for t in bounds_list):
                raise ValueError()
            return bounds_list
        except (ValueError, SyntaxError):
            raise ValueError("Invalid format for boundaries. Use comma-separated tuples, e.g., (0, 1), (10, 100)")

    def _browse_analysis_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.analysis_folder_var.set(folder_selected)

    def _get_worksheet(self):
        """Connects to Google Sheets and returns the worksheet object."""
        try:
            if self.client is None:
                self.suggestion_var.set("Connecting to Google Sheets...")
                self.root.update_idletasks()
                cred_path = self._resolve_path(self.cred_path_var.get())
                token_path = self._resolve_path(self.token_path_var.get())
                self.client = aff.authorize_gspread(cred_path, token_path)

            url = self.url_var.get()
            sheet_name = self.sheet_var.get()
            spreadsheet = self.client.open_by_url(url)
            worksheet = aff.access_worksheet(sheet_name, spreadsheet)
            return worksheet
        except Exception as e:
            messagebox.showerror("Google Sheets Error", f"Failed to connect to worksheet: {e}")
            if isinstance(e, gspread.exceptions.GSpreadException):
                self.client = None  # Reset client on GSpread error
            return None

    def append_to_sheet(self, auto=False):
        if self.last_suggestion is None:
            messagebox.showwarning("No Suggestion", "Please run the optimizer to generate a suggestion first.")
            return

        try:
            # --- Call Prerequisite Function ---
            prereq_file = self.prerequisite_func_var.get()
            if prereq_file:
                if not self._execute_user_function(prereq_file, "run_prerequisites", auto_mode=auto):
                    return  # Stop if prerequisite function fails

            # --- New EPICS Update Logic ---
            if self.update_epics_var.get():
                epics_updates = {}
                try:
                    optimizer_params_path = self._resolve_path('optimizer_params.json')
                    with open(optimizer_params_path, 'r') as f:
                        optimizer_params = json.load(f)

                    for key, value in self.last_suggestion.items():
                        if key in optimizer_params:
                            epics_pv = optimizer_params[key]
                            epics_updates[epics_pv] = value
                except FileNotFoundError:
                    messagebox.showerror("JSON Error", f"Could not find optimizer_params.json at: {optimizer_params_path}")
                    return
                except json.JSONDecodeError as e:
                    messagebox.showerror("JSON Error", f"Error decoding JSON file: {e}")
                    return

                if epics_updates:
                    if not auto:  # Manual mode, ask for confirmation
                        update_message = "The following EPICS PVs will be updated:\n\n"
                        for pv, val in epics_updates.items():
                            update_message += f"{pv}  ->  {val:.4f}\n"
                        update_message += "\nDo you want to proceed?"
                        if not messagebox.askyesno("Confirm EPICS Update", update_message):
                            messagebox.showinfo("Cancelled", "EPICS update cancelled. Nothing was changed.")
                            return  # Stop if user cancels

                    # Proceed with update (either auto mode, or manual mode confirmed)
                    try:
                        for pv, val in epics_updates.items():
                            signal = EpicsSignal(pv, name=pv)
                            signal.wait_for_connection(timeout=2.0)
                            signal.put(val)

                        if not auto:
                            messagebox.showinfo("EPICS Update", "Successfully updated EPICS PVs.")
                        else:
                            print("Auto-Optimization: Successfully updated EPICS PVs.")
                            self.autoupload_status_var.set("Status: EPICS PVs updated. Waiting for next file.")

                    except TimeoutError as e:
                        error_msg = f"Connection timed out for a PV: {e}"
                        if not auto:
                            messagebox.showerror("EPICS Error", error_msg)
                        else:
                            print(f"Auto-Optimization ERROR: {error_msg}")
                            self.autoupload_status_var.set(f"Status: EPICS Error - {error_msg}")
                        return
                    except Exception as e:
                        error_msg = f"Failed to update EPICS PVs: {e}"
                        if not auto:
                            messagebox.showerror("EPICS Error", error_msg)
                        else:
                            print(f"Auto-Optimization ERROR: {error_msg}")
                            self.autoupload_status_var.set(f"Status: EPICS Error - {error_msg}")
                        return

            # --- Existing Google Sheet Logic ---
            if self.append_to_sheet_var.get():
                worksheet = self._get_worksheet()
                if worksheet is None:
                    return  # Error already shown by helper

                headers = worksheet.row_values(1)
                if not headers:
                    headers = list(self.last_suggestion.keys())
                    worksheet.append_row(headers)

                new_row_dict = {header: "" for header in headers}
                for key, value in self.last_suggestion.items():
                    if key in new_row_dict:
                        new_row_dict[key] = value

                if self.shot_num_col_var.get() in new_row_dict and new_row_dict[self.shot_num_col_var.get()] == "":
                    # Auto-assign next shot number
                    existing_shots = worksheet.col_values(headers.index(self.shot_num_col_var.get()) + 1)[1:]
                    existing_shots_numeric = [int(s) for s in existing_shots if s.isdigit()]
                    next_shot_num = max(existing_shots_numeric, default=0) + 1
                    new_row_dict[self.shot_num_col_var.get()] = next_shot_num

                values_to_append = [new_row_dict.get(h, "") for h in headers]
                worksheet.append_row(values_to_append, value_input_option='USER_ENTERED')

                if not auto:
                    messagebox.showinfo("Success", "Successfully appended the suggestion to the sheet.")
                else:
                    print(f"Auto-Optimization: Appended new parameters to sheet for shot #{new_row_dict[self.shot_num_col_var.get()]}.")

        except Exception as e:
            messagebox.showerror("Append Error", f"An error occurred while appending: {e}")

    def run_optimization(self):
        try:
            # --- 1. Get UI inputs and worksheet ---
            input_cols = self._parse_str_list(self.inputs_var.get())
            bounds_list = self._parse_bounds(self.bounds_var.get())
            output_cols = self._parse_str_list(self.outputs_var.get())
            obj_name = self.obj_name_var.get()
            obj_type = self.obj_type_var.get().lower()
            obj_formula = self.obj_formula_var.get()
            phase = self.phase_var.get()
            local_frac = float(self.local_frac_var.get())

            if obj_type not in ["min", "max"]:
                raise ValueError("Objective type must be 'min' or 'max'")
            if len(input_cols) != len(bounds_list):
                raise ValueError(f"Mismatch: {len(input_cols)} inputs but {len(bounds_list)} boundaries defined.")

            worksheet = self._get_worksheet()
            if worksheet is None:
                return # Error already shown by helper

            # --- 2. Process Data ---
            self.suggestion_var.set("Processing data...")
            self.root.update_idletasks()
            df = pd.DataFrame(worksheet.get_all_records())
            all_needed_cols = input_cols + output_cols
            for col in all_needed_cols:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in the sheet.")
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Calculate objective
            def calculate_obj(row):
                y = row[output_cols].values
                try:
                    return eval(obj_formula, {"__builtins__": {}}, {"y": y, "np": np})
                except Exception as e:
                    raise ValueError(f"Error evaluating objective formula: {e}")

            df[obj_name] = df.apply(calculate_obj, axis=1)
            
            
            # --- Write calculated objective back to Google Sheet ---
            self.suggestion_var.set("Updating Google Sheet with objective values...")
            self.root.update_idletasks()

            sheet_headers = worksheet.row_values(1)
            obj_col_index = -1
            try:
                obj_col_index = sheet_headers.index(obj_name) + 1 # 1-based index
            except ValueError:
                # If objective column doesn't exist, add it to headers
                worksheet.append_row([obj_name])
                sheet_headers = worksheet.row_values(1) # Re-read headers
                obj_col_index = sheet_headers.index(obj_name) + 1

            cells_to_update = []
            # Iterate through the DataFrame and prepare cells for update
            # df.index is 0-based, sheet rows are 1-based, plus 1 for header row
            for df_idx, obj_value in df[obj_name].items():
                if pd.notna(obj_value): # Only update if objective is not NaN
                    sheet_row = df_idx + 2 # +1 for 1-based, +1 for header
                    cells_to_update.append(gspread.Cell(row=sheet_row, col=obj_col_index, value=str(obj_value)))
            
            if cells_to_update:
                worksheet.update_cells(cells_to_update)
                self.suggestion_var.set("Google Sheet updated with objective values.")
            else:
                self.suggestion_var.set("No objective values to update in Google Sheet.")

            # --- 3. Prepare for Optimizer ---
            input_names = dict(zip(input_cols, bounds_list))

            # --- 4. Run Optimizer ---
            self.suggestion_var.set(f"Running Optimization (Phase: {phase})...")
            self.root.update_idletasks()
            
            next_params_list = bo.suggest_next_params_bo(
                df, input_names, obj_name, obj_type, phase=phase, local_frac=local_frac
            )
            
            self.last_suggestion = dict(zip(input_names.keys(), next_params_list))
            
            suggestion_str = "Next Suggested Params:\n"
            suggestion_str += ", ".join([f"{k}={v:.4f}" for k, v in self.last_suggestion.items()])
            self.suggestion_var.set(suggestion_str)

            # --- 5. Find and Display Current Best ---
            df_clean = df.dropna(subset=all_needed_cols + [obj_name])
            if not df_clean.empty:
                if obj_type == "max":
                    best_row = df_clean.loc[df_clean[obj_name].idxmax()]
                else:
                    best_row = df_clean.loc[df_clean[obj_name].idxmin()]

                best_str = "Current Best:\n"
                best_str += f"Row Index: {best_row.name}, Objective ({obj_name}): {best_row[obj_name]:.4f}\n"
                best_str += "Inputs: " + ", ".join([f"{k}={best_row[k]:.4f}" for k in input_cols])
                self.best_var.set(best_str)
            else:
                self.best_var.set("No valid historical data to determine a best result.")

        except Exception as e:
            self.suggestion_var.set("An error occurred.")
            self.best_var.set("")
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizerGUI(root)
    root.mainloop()
