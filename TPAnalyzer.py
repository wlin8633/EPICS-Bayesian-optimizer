import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import time
import threading
import numpy as np
from PIL import Image
import json
import re
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from ImageMonitor import MonitorWindow
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import pathlib

# --- Analysis functions from TPanalysis.py ---

def x2E(x_dis, m_amu, qom_eV_amu, By, lB, dB):
    """
    x_dis: x distance in meter. Unit: m
    """
    eV = 1.6e-19
    amu = 1.66e-27 / 1.007276466812
    cB = lB * (lB / 2 + dB) * By
    # Handle potential division by zero if x_dis is close to zero
    with np.errstate(divide='ignore', invalid='ignore'):
        energy_joules = 1/2 * m_amu * amu * (qom_eV_amu * (eV / amu) / x_dis * cB)**2
    energy_MeV = energy_joules / (1e6 * eV)
    energy_MeV[np.isinf(energy_MeV)] = 0 # Replace inf with 0
    return energy_MeV

def analyze_energy_spectrum(raw_image, energy_map, bbox):
    """
    raw_image: 2D numpy array of the raw image
    energy_map: 2D numpy array of the energy map
    bbox: tuple of (x_min, y_min, x_max, y_max)
    
    returns: energy_bins, spectrum
    """
    x_min, y_min, x_max, y_max = bbox
    roi_image = raw_image[y_min:y_max, x_min:x_max]
    roi_energy = energy_map[y_min:y_max, x_min:x_max]
    
    # Flatten arrays
    roi_image_flat = roi_image.flatten()
    roi_energy_flat = roi_energy.flatten()
    
    # Create histogram
    energy_bins = np.linspace(np.nanmin(roi_energy_flat), np.nanmax(roi_energy_flat), 100)
    spectrum, _ = np.histogram(roi_energy_flat, bins=energy_bins, weights=roi_image_flat)
    
    return energy_bins[:-1], spectrum

def plot_spectrum(energy_bins, spectrum):
    fig, ax = plt.subplots()
    ax.plot(energy_bins, spectrum)
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel('Counts')
    ax.set_title('Energy Spectrum')
    plt.show()
    
def save_spectrum(filename, energy_bins, spectrum, std_spectrum=None):
    """Saves the energy spectrum to a text file."""
    if std_spectrum is not None:
        header = "Energy (MeV), Mean_Counts, STD_Counts"
        data = np.c_[energy_bins, spectrum, std_spectrum]
    else:
        header = "Energy (MeV), Counts"
        data = np.c_[energy_bins, spectrum]
    np.savetxt(filename, data, header=header, delimiter=',')


from scipy.spatial import cKDTree

class TPAnalyzerWindow:
    def __init__(self, master, monitor_window=None):
        self.master = master
        self.monitor_window = monitor_window
        self.window = tk.Toplevel(master)
        self.window.title("TP Analyzer")
        self.window.geometry("1200x800")

        self.script_dir = pathlib.Path(__file__).parent
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        self.seen_files = set()
        self.accumulated_images = []
        self.accumulated_image_paths = [] # To store file paths
        self.state_file = "tp_analyzer_state.json"
        self.last_mean_img = None
        self.last_std_img = None
        self.last_tp_count_roi = None
        self.last_roi_coords = None
        self.analysis_results_window = None

        self._setup_widgets()
        self.load_state()
        self.periodic_save()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _resolve_path(self, path_str):
        if not path_str or os.path.isabs(path_str):
            return path_str
        return str(self.script_dir / path_str)

    def _setup_widgets(self):
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = ttk.Frame(main_frame, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Controls in Left Frame ---
        source_frame = ttk.LabelFrame(left_frame, text="Input Source", padding="10")
        source_frame.pack(fill=tk.X, pady=5)

        ttk.Label(source_frame, text="Source Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.path_var = tk.StringVar()
        path_entry = ttk.Entry(source_frame, textvariable=self.path_var, width=40)
        path_entry.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E))
        
        browse_folder_button = ttk.Button(source_frame, text="Folder", command=self._browse_folder)
        browse_folder_button.grid(row=1, column=1, sticky=tk.W, padx=2)
        browse_batch_button = ttk.Button(source_frame, text="Batch Select", command=self._browse_batch_files)
        browse_batch_button.grid(row=1, column=2, sticky=tk.W, padx=2)

        ttk.Label(source_frame, text="Images to Accumulate:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.accum_num_var = tk.StringVar(value="10")
        ttk.Entry(source_frame, textvariable=self.accum_num_var, width=10).grid(row=2, column=1, sticky=tk.W)

        update_roi_button = ttk.Button(source_frame, text="Update ROI/Zero", command=self.update_roi_from_monitor)
        update_roi_button.grid(row=2, column=2, sticky=tk.W, padx=5)
        
        self.auto_analysis_var = tk.BooleanVar(value=False)
        auto_analysis_check = ttk.Checkbutton(source_frame, text="Auto Analysis Loop", variable=self.auto_analysis_var)
        auto_analysis_check.grid(row=3, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(5,0))

        source_frame.columnconfigure(1, weight=1)

        analysis_frame = ttk.LabelFrame(left_frame, text="Analysis Parameters", padding="10")
        analysis_frame.pack(fill=tk.X, pady=5)

        self.param_vars = {}
        params = {
            "Z": "1", "A": "1", "pixel_size": "2.4e-5", "zero_width": "5",
            "Bz": "0.2", "lB": "0.1", "dB": "0.1",
            "Ez": "0", "lE": "0", "dE": "0",
            "minE": "1", "maxE": "20", "binE": "1000",
            "is_gaussian_filter": "True", "gaussian_sigma": "2.0"
        }
        
        col = 0
        row = 0
        for name, val in params.items():
            ttk.Label(analysis_frame, text=f"{name}:").grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(value=val)
            self.param_vars[name] = var
            ttk.Entry(analysis_frame, textvariable=var, width=12).grid(row=row, column=col+1, sticky=tk.W, padx=5, pady=2)
            col += 2
            if col >= 4:
                col = 0
                row += 1

        save_frame = ttk.LabelFrame(left_frame, text="Save Options", padding="10")
        save_frame.pack(fill=tk.X, pady=5)

        self.save_spectrum_var = tk.BooleanVar(value=True)
        save_check = ttk.Checkbutton(save_frame, text="Save Analysis Data", variable=self.save_spectrum_var)
        save_check.grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5)

        ttk.Label(save_frame, text="Save Prefix:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.save_prefix_var = tk.StringVar()
        ttk.Entry(save_frame, textvariable=self.save_prefix_var, width=40).grid(row=1, column=1, sticky=(tk.W, tk.E))

        ttk.Label(save_frame, text="Save Folder:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.save_path_var = tk.StringVar()
        save_path_entry = ttk.Entry(save_frame, textvariable=self.save_path_var, width=40)
        save_path_entry.grid(row=2, column=1, sticky=(tk.W, tk.E))
        browse_save_button = ttk.Button(save_frame, text="Browse...", command=self._browse_save_folder)
        browse_save_button.grid(row=3, column=1, sticky=tk.W, padx=5)
        save_frame.columnconfigure(1, weight=1)

        action_frame = ttk.Frame(left_frame, padding="10")
        action_frame.pack(fill=tk.X, pady=10)

        self.start_button = tk.Button(action_frame, text="Start/Preview", command=self.start_accumulation_or_preview, bg="blue", fg="white", height=2)
        self.start_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.stop_button = tk.Button(action_frame, text="Stop", command=self.stop_accumulation, bg="orange", fg="white", height=2, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.run_button = tk.Button(action_frame, text="Confirm & Run Analysis", command=lambda: self.confirm_and_run_analysis(), bg="green", fg="white", height=2, state=tk.DISABLED)
        self.run_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Status: Idle")
        ttk.Label(left_frame, textvariable=self.status_var, wraplength=380).pack(pady=10, fill=tk.X)

        # --- Preview Canvas in Right Frame ---
        preview_frame = ttk.LabelFrame(right_frame, text="Analysis Preview", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True)

        self.preview_fig = Figure(dpi=100)
        self.preview_ax = self.preview_fig.add_subplot(111)
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, master=preview_frame)
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.preview_fig.tight_layout()

    def _browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.path_var.set(folder_selected)
            self.start_button.config(text="Start Accumulation")
            self.run_button.config(state=tk.DISABLED)

    def _browse_batch_files(self):
        files = filedialog.askopenfilenames(
            title="Select image files for analysis",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        if not files:
            return

        # If only one file is selected, treat it as a single-file preview setup
        if len(files) == 1:
            self.path_var.set(files[0])
            self.start_button.config(text="Preview Single Image")
            self.run_button.config(state=tk.DISABLED)
            return

        # If multiple files are selected, confirm and run batch analysis
        if not messagebox.askyesno("Confirm Batch Analysis", f"This will analyze {len(files)} files one-by-one and save an output for each. This may take time. Continue?"):
            return
            
        # Since this can take a long time, run it in a separate thread
        # to avoid freezing the GUI.
        processing_thread = threading.Thread(
            target=self._run_sequential_analysis, 
            args=(files,), 
            daemon=True
        )
        processing_thread.start()

    def _run_sequential_analysis(self, files):
        """
        This method runs in a thread to process a list of files sequentially.
        """
        for i, file_path in enumerate(files):
            try:
                self.status_var.set(f"Processing {i+1}/{len(files)}: {os.path.basename(file_path)}")

                # 1. Load the image
                img = np.array(Image.open(file_path))

                # 2. Set the data for a single-image analysis
                self.last_mean_img = img
                self.last_std_img = np.zeros_like(img)
                
                # 3. We need to run the preview calculations to generate the analysis mask,
                # but we don't need to redraw the GUI every time.
                # The existing _draw_preview does both. For now, we call it, accepting
                # that the GUI will flicker for each file.
                self._draw_preview()

                # 4. Run the final analysis in non-interactive mode
                self.confirm_and_run_analysis(image_paths_for_analysis=[file_path], is_batch_mode=True)
                
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
                self.status_var.set(f"Error on file {i+1}. See console for details.")
                time.sleep(2) # Give user time to see error
        
        self.status_var.set(f"Batch processing of {len(files)} files complete.")

    def _browse_save_folder(self):
        folder_selected = filedialog.askdirectory(
            title="Select folder to save analysis data"
        )
        if folder_selected:
            self.save_path_var.set(folder_selected)

    def update_roi_from_monitor(self):
        """
        Fetches the latest ROI and Zero Point from the monitor window and redraws the preview.
        """
        if self.last_mean_img is None:
            messagebox.showinfo("Info", "Please load an image first before updating ROI.")
            return
        
        self.status_var.set("Updating ROI/Zero from monitor and redrawing preview...")
        self.window.update_idletasks()
        self._draw_preview()
        self.status_var.set("Preview updated with new ROI/Zero.")

    def start_accumulation_or_preview(self):
        source_path = self._resolve_path(self.path_var.get())
        if not source_path:
            messagebox.showerror("Error", "Please select a source path.")
            return

        self.run_button.config(state=tk.DISABLED)

        # If auto-analysis is on, use the new loop
        if self.auto_analysis_var.get():
            if not os.path.isdir(source_path):
                messagebox.showerror("Error", "Auto-analysis only works with a folder source.")
                return
            
            self.accumulated_images = [] # Initialize image buffer
            self.stop_event.clear()
            
            # Initialize seen_files with current directory contents
            try:
                self.seen_files = {os.path.join(source_path, f) for f in os.listdir(source_path)}
            except OSError as e:
                messagebox.showerror("Error", f"Could not read folder contents: {e}")
                return

            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set("Starting auto-analysis loop...")
            self.window.after(100, self._auto_analysis_loop) # Kick off the loop
            return

        # --- Logic for loading a single file for manual preview ---
        if os.path.isfile(source_path):
            try:
                self.status_var.set(f"Loading {os.path.basename(source_path)}...")
                self.window.update_idletasks()
                
                img = np.array(Image.open(source_path))
                
                # Set up for analysis (mean of 1 is self, std is 0)
                self.last_mean_img = img
                self.last_std_img = np.zeros_like(img)
                self.accumulated_image_paths = [source_path]

                self._draw_preview() # Show the preview, user can then click 'Run Analysis'
            except Exception as e:
                messagebox.showerror("Image Load Error", f"Failed to load single image: {e}")
            return

        # --- Logic for accumulating from a folder ---
        if os.path.isdir(source_path):
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                messagebox.showinfo("Info", "Accumulation is already in progress.")
                return
            
            self.accumulated_images = []
            self.accumulated_image_paths = []
            self.stop_event.clear()
            
            try:
                self.seen_files = {os.path.join(source_path, f) for f in os.listdir(source_path)}
            except OSError as e:
                messagebox.showerror("Error", f"Could not read folder contents: {e}")
                return

            self.monitoring_thread = threading.Thread(target=self._monitor_for_accumulation, daemon=True)
            self.monitoring_thread.start()
            
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
        else:
            # This case handles when the path is a message like "X files selected..."
            if "files selected" not in source_path:
                 messagebox.showerror("Error", "Invalid source path.")

    def _auto_analysis_loop(self):
        if self.stop_event.is_set():
            self.status_var.set("Auto-analysis stopped.")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            return

        source_path = self._resolve_path(self.path_var.get())
        try:
            num_to_accumulate = int(self.accum_num_var.get())
        except ValueError:
            self.status_var.set("Status: Error - Invalid accumulation number.")
            self.stop_event.set()
            return

        # --- Core monitoring logic ---
        try:
            current_files = {os.path.join(source_path, f) for f in os.listdir(source_path)}
        except Exception as e:
            self.status_var.set(f"Error reading source folder: {e}")
            self.stop_event.set()
            return

        new_files = sorted(list(current_files - self.seen_files), key=os.path.getmtime)

        if new_files:
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
            for file_path in new_files:
                if os.path.splitext(file_path)[1].lower() in image_extensions:
                    try:
                        # Add a small delay and retry mechanism for file loading
                        for _ in range(3):
                            try:
                                img = Image.open(file_path)
                                self.accumulated_images.append(np.array(img))
                                self.accumulated_image_paths.append(file_path)
                                break # Success
                            except PermissionError:
                                time.sleep(0.5)
                        else: # If loop finishes without break
                            raise IOError(f"Could not open file after retries: {file_path}")
                    except Exception as e:
                        print(f"Error opening image {file_path}: {e}")
        
        self.seen_files = current_files
        # --- End of monitoring logic ---

        self.status_var.set(f"Auto-monitoring... Collected {len(self.accumulated_images)}/{num_to_accumulate} images.")

        # --- Batch processing logic ---
        if len(self.accumulated_images) >= num_to_accumulate:
            self.status_var.set(f"Have {len(self.accumulated_images)} images. Processing a batch of {num_to_accumulate}.")
            self.window.update_idletasks()

            batch_images = self.accumulated_images[:num_to_accumulate]
            self.accumulated_images = self.accumulated_images[num_to_accumulate:]
            
            # Keep track of the paths for the batch being processed
            batch_image_paths = self.accumulated_image_paths[:num_to_accumulate]
            self.accumulated_image_paths = self.accumulated_image_paths[num_to_accumulate:]

            # Set up the batch for analysis (mean/std)
            images_stack = np.stack(batch_images, axis=0)
            self.last_mean_img = np.mean(images_stack, axis=0)
            self.last_std_img = np.std(images_stack, axis=0)

            self._draw_preview()
            self.confirm_and_run_analysis(image_paths_for_analysis=batch_image_paths)

        # --- Schedule next run ---
        if not self.stop_event.is_set():
            self.window.after(2000, self._auto_analysis_loop) # Check for new files every 2s
        else:
            self.status_var.set("Auto-analysis stopped.")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def stop_accumulation(self):
        self.stop_event.set()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Status: Accumulation stopped by user.")

    def _monitor_for_accumulation(self):
        source_path = self._resolve_path(self.path_var.get())
        try:
            num_to_accumulate = int(self.accum_num_var.get())
        except ValueError:
            self.status_var.set("Status: Error - Invalid accumulation number.")
            return

        self.status_var.set(f"Status: Waiting for new files...")
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

        while not self.stop_event.is_set():
            try:
                current_files = {os.path.join(source_path, f) for f in os.listdir(source_path)}
                new_files = sorted(list(current_files - self.seen_files), key=os.path.getmtime)

                for file_path in new_files:
                    if os.path.splitext(file_path)[1].lower() in image_extensions:
                        try:
                            img = Image.open(file_path)
                            self.accumulated_images.append(np.array(img))
                            self.accumulated_image_paths.append(file_path)
                            self.status_var.set(f"Status: Collected {len(self.accumulated_images)}/{num_to_accumulate} images...")
                            if len(self.accumulated_images) >= num_to_accumulate:
                                self.stop_event.set() # Signal to stop
                                break
                        except Exception as e:
                            print(f"Error opening image {file_path}: {e}")
                
                self.seen_files = current_files
                if self.stop_event.is_set():
                    break
                time.sleep(1) # Check for new files every second

            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(2)

        if len(self.accumulated_images) >= num_to_accumulate:
            # Set up the batch for analysis (mean/std)
            images_to_process = self.accumulated_images[:num_to_accumulate]
            images_stack = np.stack(images_to_process, axis=0)
            self.last_mean_img = np.mean(images_stack, axis=0)
            self.last_std_img = np.std(images_stack, axis=0)
            self.accumulated_image_paths = self.accumulated_image_paths[:num_to_accumulate]

            self._draw_preview()
        
        # Reset buttons after thread finishes
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def _draw_preview(self):
        if self.last_mean_img is None:
            messagebox.showerror("Error", "No image data loaded.")
            return
        if self.monitor_window is None or not self.monitor_window.is_alive():
            messagebox.showerror("Error", "Image Monitor window is not open.")
            return

        try:
            config = {name: var.get() for name, var in self.param_vars.items()}
            for k in ["Z", "A", "binE"]: config[k] = int(float(config[k]))
            for k in ["pixel_size", "zero_width", "Bz", "lB", "dB", "Ez", "lE", "dE", "minE", "maxE"]: config[k] = float(config[k])
            
            config["zero_pt"] = self.monitor_window.get_zero_point()

            roi_bbox = self.monitor_window.get_roi_bbox()
            if config["zero_pt"] is None or roi_bbox is None:
                raise ValueError("Zero Point or ROI not set in Image Monitor.")

            x_min, y_min, x_max, y_max = roi_bbox
            tp_raw_roi = self.last_mean_img[y_min:y_max, x_min:x_max]
            self.last_roi_coords = {'x_min': x_min, 'y_min': y_min}

            yy_roi, xx_roi = np.mgrid[y_min:y_max, x_min:x_max]
            x_dis = (xx_roi - config["zero_pt"][0]) * config["pixel_size"]
            y_dis = (yy_roi - config["zero_pt"][1]) * config["pixel_size"]
            beamWidth = config["zero_width"] * config["pixel_size"]

            m = config["A"] * 1.66e-27
            q = config["Z"] * 1.6e-19
            theo_proton_E = np.linspace(config["minE"], config["maxE"], config["binE"]) * 1e6 * 1.6e-19
            with np.errstate(divide='ignore'):
                theo_proton_v = (2 * theo_proton_E / m)**0.5
            
            theo_proton_TP_x = (q*config["Bz"])/(m*theo_proton_v) * config["lB"]*(config["lB"]/2 + config["dB"])
            theo_proton_TP_y = (q*config["Ez"])/(m*theo_proton_v**2) * config["lE"]*(config["lE"]/2 + config["dE"])

            parabola_points = np.c_[theo_proton_TP_x, -theo_proton_TP_y]
            roi_grid_points = np.c_[x_dis.ravel(), y_dis.ravel()]
            
            tree = cKDTree(parabola_points)
            min_dist_sq, _ = tree.query(roi_grid_points, k=1, p=2)
            min_dist_sq = min_dist_sq**2

            tp_mask = min_dist_sq.reshape(x_dis.shape) < beamWidth**2
            self.last_tp_count_roi = tp_raw_roi.astype(np.float32) * tp_mask

            # --- Drawing Part ---
            self.preview_ax.clear()
            self.preview_ax.imshow(self.last_mean_img, cmap='gray', aspect='equal')
            
            masked_tp_count = np.ma.masked_where(self.last_tp_count_roi == 0, self.last_tp_count_roi)
            self.preview_ax.imshow(masked_tp_count, cmap='viridis', 
                                   extent=[x_min, x_max, y_max, y_min], aspect='equal')

            self.preview_ax.plot(config["zero_pt"][0], config["zero_pt"][1], 'r+', markersize=12, markeredgewidth=2, label="Zero Point")
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor='lime', facecolor='none', label="Analysis ROI")
            self.preview_ax.add_patch(rect)

            self.preview_ax.set_title("Preview: Masked Data Overlaid on Full Image")
            self.preview_ax.legend()
            self.preview_fig.tight_layout()
            self.preview_canvas.draw()
            self.status_var.set("Preview updated. Ready to run analysis.")
            self.run_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Preview Error", f"Could not generate preview: {e}")
            self.status_var.set(f"Status: Preview Error - {e}")

    def confirm_and_run_analysis(self, image_paths_for_analysis=None, is_batch_mode=False):
        self.status_var.set("Starting final analysis...")
        self.window.update_idletasks()
        
        if self.last_tp_count_roi is None or self.last_std_img is None:
            messagebox.showerror("Error", "No preview data available. Click 'Start/Preview' first.")
            return

        # If paths are not passed from an auto-loop, use the instance's stored paths
        if image_paths_for_analysis is None:
            image_paths_for_analysis = self.accumulated_image_paths

        if not image_paths_for_analysis:
            messagebox.showerror("Error", "No image file paths were found for this analysis.")
            return

        try:
            # --- Shot Number Extraction ---
            last_image_path = image_paths_for_analysis[-1]
            shot_number_match = re.search(r'\d+', os.path.basename(last_image_path))
            if shot_number_match:
                shot_number = int(shot_number_match.group(0))
            else:
                shot_number = -1 # Default/error value

            config = {name: var.get() for name, var in self.param_vars.items()}
            for k in ["Z", "A", "binE"]: config[k] = int(float(config[k]))
            for k in ["pixel_size", "zero_width", "Bz", "lB", "dB", "Ez", "lE", "dE", "minE", "maxE", "gaussian_sigma"]: config[k] = float(config[k])
            config["is_gaussian_filter"] = config["is_gaussian_filter"].lower() in ['true', '1', 't', 'y', 'yes']
            config["filePath"] = self.path_var.get()
            config["zero_pt"] = self.monitor_window.get_zero_point()
            config["roi_bbox"] = self.monitor_window.get_roi_bbox()
            config["bg_bbox"] = self.monitor_window.get_bg_bbox()
            
            tp_count_roi = self.last_tp_count_roi
            mean_spectrum = np.sum(tp_count_roi, axis=0)
            
            # Calculate std spectrum from std image
            roi_bbox = self.monitor_window.get_roi_bbox()
            x_min, y_min, x_max, y_max = roi_bbox
            std_roi = self.last_std_img[y_min:y_max, x_min:x_max]
            # The variance of the sum is the sum of variances
            std_spectrum = np.sqrt(np.sum(std_roi**2, axis=0))

            roi_x_pixels = np.arange(self.last_roi_coords['x_min'], self.last_roi_coords['x_min'] + tp_count_roi.shape[1])
            x_dis_row = (roi_x_pixels - config["zero_pt"][0]) * config["pixel_size"]

            m = config["A"] * 1.66e-27
            q = config["Z"] * 1.6e-19
            D_B = config["dB"] + config["lB"] / 2.0
            
            with np.errstate(divide='ignore', invalid='ignore'):
                energy_J = (q**2 * config["Bz"]**2 * config["lB"]**2 * D_B**2) / (2 * m * x_dis_row**2)
            ene_from_x = energy_J / 1.602e-13

            valid_indices = x_dis_row > 0
            mean_spectrum = mean_spectrum[valid_indices]
            std_spectrum = std_spectrum[valid_indices]
            ene_from_x = ene_from_x[valid_indices]
            x_dis_filter = x_dis_row[valid_indices]

            valid_indices = ene_from_x < config["maxE"]
            es_y_mean = mean_spectrum[valid_indices]
            es_y_std = std_spectrum[valid_indices]
            es_x = ene_from_x[valid_indices]
            es_dis = x_dis_filter[valid_indices]

            unsmoothed_mean_spectrum = es_y_mean.copy()
            unsmoothed_std_spectrum = es_y_std.copy()

            if config["is_gaussian_filter"]:
                smoothed_mean_spectrum = gaussian_filter(es_y_mean, sigma=config["gaussian_sigma"])
                # Propagate error for smoothing (approximation)
                smoothed_std_spectrum = gaussian_filter(es_y_std**2, sigma=config["gaussian_sigma"])**0.5
            else:
                smoothed_mean_spectrum = unsmoothed_mean_spectrum
                smoothed_std_spectrum = unsmoothed_std_spectrum
            
            if not is_batch_mode:
                self._show_analysis_results(tp_count_roi, es_x, es_dis, smoothed_mean_spectrum, smoothed_std_spectrum)

            if self.save_spectrum_var.get():
                save_folder = self._resolve_path(self.save_path_var.get())
                
                # If no save folder is specified, default to the source image's directory
                if not save_folder or not os.path.isdir(save_folder):
                    save_folder = os.path.dirname(last_image_path)

                os.makedirs(save_folder, exist_ok=True)
                    
                prefix = self.save_prefix_var.get()
                base_filename = os.path.splitext(os.path.basename(last_image_path))[0]
                
                # Combine prefix and filename
                if prefix:
                    final_prefix = f"{prefix}_{base_filename}"
                else:
                    final_prefix = base_filename

                filename = f"{final_prefix}_tp_analysis.npz"
                npz_path = os.path.join(save_folder, filename)

                # Update config for saving
                config["savFolderPath"] = save_folder

                np.savez(npz_path, 
                            config=config,
                            shot_number=shot_number,
                            avg_image=self.last_mean_img,
                            std_image=self.last_std_img,
                            energy_axis=es_x, 
                            distance_axis=es_dis,
                            mean_spectrum=smoothed_mean_spectrum,
                            std_spectrum=smoothed_std_spectrum)
                
                # Update status without overwriting the main batch status
                if not is_batch_mode:
                    self.status_var.set(f"Analysis finished. Results saved to {npz_path}")
                    
            elif not is_batch_mode:
                self.status_var.set("Thomson Parabola analysis finished.")

        except Exception as e:
            if not is_batch_mode:
                messagebox.showerror("Analysis Error", f"An error occurred during analysis: {e}")
            self.status_var.set(f"Status: Analysis Error - {e}")

    def _show_analysis_results(self, tp_count_roi, es_x, es_dis, mean_spectrum, std_spectrum):
        # If window doesn't exist or has been closed by the user, create it
        if not hasattr(self, 'analysis_results_window') or not self.analysis_results_window or not self.analysis_results_window.winfo_exists():
            self.analysis_results_window = tk.Toplevel(self.window)
            self.analysis_results_window.title("Analysis Results")
            self.analysis_results_window.geometry("1200x500")

            self.results_fig = Figure(figsize=(12, 5), dpi=100)
            # Store axes in a list for easy clearing and access
            self.results_axes = [
                self.results_fig.add_subplot(131),
                self.results_fig.add_subplot(132),
                self.results_fig.add_subplot(133)
            ]
            
            self.results_canvas = FigureCanvasTkAgg(self.results_fig, master=self.analysis_results_window)
            self.results_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Clear previous plots from all axes
        for ax in self.results_axes:
            ax.clear()

        # Get the axes back for plotting
        ax1, ax2, ax3 = self.results_axes

        # Plot new data
        ax1.imshow(tp_count_roi, cmap='gray_r', aspect='auto')
        ax1.set_xlabel("x (pixels in ROI)")
        ax1.set_ylabel("y (pixels in ROI)")
        ax1.set_title("Processed Data in ROI")

        ax2.plot(es_x, mean_spectrum, 'b-', label='Mean Spectrum')
        ax2.fill_between(es_x, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, color='blue', alpha=0.3, label='Std. Dev.')
        ax2.set_xlabel("Energy (MeV)")
        ax2.set_ylabel("Counts")
        ax2.set_title("Energy Spectrum")
        ax2.legend()
        ax2.grid(True)

        ax3.plot(es_dis, mean_spectrum, 'b-')
        ax3.fill_between(es_dis, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, color='blue', alpha=0.3)
        ax3.set_xlabel("Distance (m)")
        ax3.set_ylabel("Counts")
        ax3.set_title("Spatial Spectrum")
        ax3.grid(True)

        self.results_fig.tight_layout()
        self.results_canvas.draw()
        
        # Bring the window to the front without aggressively stealing focus
        self.analysis_results_window.lift()

    def load_state(self):
        default_params = {
            "Z": "1", "A": "1", "pixel_size": "2.4e-5", "zero_width": "5",
            "Bz": "0.2", "lB": "0.1", "dB": "0.1",
            "Ez": "0", "lE": "0", "dE": "0",
            "minE": "1", "maxE": "20", "binE": "1000",
            "is_gaussian_filter": "True", "gaussian_sigma": "2.0"
        }
        default_state = {
            "source_path": "",
            "accum_num": "10",
            "params": default_params,
            "save_spectrum": True,
            "save_path": "",
            "save_prefix": "",
            "auto_analysis": False
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

        self.path_var.set(state.get("source_path", default_state["source_path"]))
        self.accum_num_var.set(state.get("accum_num", default_state["accum_num"]))
        self.auto_analysis_var.set(state.get("auto_analysis", default_state["auto_analysis"]))

        params_to_load = state.get("params", default_params)
        for name, var in self.param_vars.items():
            var.set(params_to_load.get(name, default_params.get(name, "")))
        
        self.save_spectrum_var.set(state.get("save_spectrum", default_state["save_spectrum"]))
        self.save_path_var.set(state.get("save_path", default_state["save_path"]))
        self.save_prefix_var.set(state.get("save_prefix", default_state.get("save_prefix", "")))

    def save_state(self):
        state = {
            "source_path": self.path_var.get(),
            "accum_num": self.accum_num_var.get(),
            "params": {name: var.get() for name, var in self.param_vars.items()},
            "save_spectrum": self.save_spectrum_var.get(),
            "save_path": self.save_path_var.get(),
            "save_prefix": self.save_prefix_var.get(),
            "auto_analysis": self.auto_analysis_var.get()
        }
        try:
            state_path = self._resolve_path(self.state_file)
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=4)
        except IOError as e:
            print(f"Error saving TP analyzer state: {e}")

    def periodic_save(self):
        self.save_state()
        self.window.after(30000, self.periodic_save)

    def on_closing(self):
        self.save_state()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_event.set()
            self.monitoring_thread.join(timeout=1)
        self.window.destroy()

    def lift(self):
        self.window.lift()

    def is_alive(self):
        return self.window.winfo_exists()

if __name__ == '__main__':
    # Example of how to run this window standalone
    root = tk.Tk()
    root.title("Main App")
    # Hide the dummy root window, as the Toplevels are the main interface
    root.withdraw()

    # The Analyzer depends on the Monitor, so create the Monitor window first.
    monitor_app = MonitorWindow(root)
    
    # Now create the analyzer and pass the monitor instance to it.
    analyzer_app = TPAnalyzerWindow(root, monitor_window=monitor_app)

    # Lift the windows to ensure they are visible
    monitor_app.lift()
    analyzer_app.lift()
    
    root.mainloop()
