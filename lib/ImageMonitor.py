import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import time
import threading
from queue import Queue, Empty
from PIL import Image, ImageTk
import numpy as np
import json

import matplotlib.patches as patches

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class MonitorWindow:
    def __init__(self, master):
        self.master = master
        self.window = tk.Toplevel(master)
        self.window.title("Image Monitor")
        self.window.geometry("800x750")

        self.image_queue = Queue()
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        self.last_file_path = None
        self.background_image = None
        self.bg_factor = 1.0 # New: Background scaling factor
        self.state_file = "monitor_state.json"
        self.last_img_array = None
        self.last_filename = None

        # Analysis selection attributes
        self.edit_mode = False
        self.zero_pt = None
        self.roi_bbox = None
        self.start_x = None
        self.start_y = None
        self.original_title = "Image Monitor"
        self.roi_rect_artist = None
        self.zero_pt_artist = None
        self.drag_rect_artist = None

        self._setup_widgets()
        self.load_state()
        self.process_queue()
        self.periodic_save()
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.bind("<Control-e>", self.toggle_edit_mode)

    def toggle_edit_mode(self, event=None):
        self.edit_mode = not self.edit_mode
        if self.edit_mode:
            self.window.title(f"{self.original_title} - EDIT MODE (Ctrl+E to exit)")
            self.toolbar.set_message("EDIT MODE: Left-click to set zero point, drag to set ROI.")
            # Disable matplotlib's default pan/zoom bindings
            self.toolbar.children['!button2'].config(state=tk.DISABLED)
            self.toolbar.children['!button3'].config(state=tk.DISABLED)
        else:
            self.window.title(self.original_title)
            self.toolbar.set_message("")
            # Re-enable matplotlib's default pan/zoom bindings
            self.toolbar.children['!button2'].config(state=tk.NORMAL)
            self.toolbar.children['!button3'].config(state=tk.NORMAL)
        print(f"Edit mode {'enabled' if self.edit_mode else 'disabled'}.")

    def on_canvas_click(self, event):
        if self.edit_mode and event.inaxes == self.ax:
            self.start_x = event.xdata
            self.start_y = event.ydata
            
            if self.drag_rect_artist:
                self.drag_rect_artist.remove()
            self.drag_rect_artist = self.ax.add_patch(
                patches.Rectangle((self.start_x, self.start_y), 0, 0, fill=False, color='lime', linestyle='--', linewidth=1)
            )
            self.canvas.draw()

    def on_canvas_drag(self, event):
        if self.edit_mode and event.inaxes == self.ax and self.start_x is not None:
            width = event.xdata - self.start_x
            height = event.ydata - self.start_y
            
            self.drag_rect_artist.set_width(width)
            self.drag_rect_artist.set_height(height)

            # Handle dragging in any direction
            if width < 0:
                self.drag_rect_artist.set_x(event.xdata)
            if height < 0:
                self.drag_rect_artist.set_y(event.ydata)
            
            self.canvas.draw()

    def on_canvas_release(self, event):
        if self.edit_mode and self.start_x is not None:
            if self.drag_rect_artist:
                self.drag_rect_artist.remove()
                self.drag_rect_artist = None

            end_x = event.xdata
            end_y = event.ydata

            if end_x is None or end_y is None: # Released outside of axes
                self.start_x, self.start_y = None, None
                self.canvas.draw()
                return

            # Check if it was a click (very short drag)
            # Use a threshold based on pixels on screen to be robust against zoom
            display_dx = abs(event.x - self.canvas.get_tk_widget().winfo_pointerx())
            display_dy = abs(event.y - self.canvas.get_tk_widget().winfo_pointery())

            if abs(self.start_x - end_x) < 5 and abs(self.start_y - end_y) < 5:
                self.zero_pt = (int(self.start_x), int(self.start_y))
                print(f"Zero point set to: {self.zero_pt}")
            else: # It was a drag, so set bbox
                self.roi_bbox = (
                    int(min(self.start_x, end_x)), int(min(self.start_y, end_y)),
                    int(max(self.start_x, end_x)), int(max(self.start_y, end_y))
                )
                print(f"ROI bbox set to: {self.roi_bbox}")

            self.start_x, self.start_y = None, None
            self.draw_overlays()
            self.canvas.draw()

    def draw_overlays(self):
        # Delete previous drawings
        if self.zero_pt_artist:
            try:
                self.zero_pt_artist.remove()
            except (ValueError, AttributeError): # Handle cases where artist is already gone
                pass
            self.zero_pt_artist = None
        if self.roi_rect_artist:
            try:
                self.roi_rect_artist.remove()
            except (ValueError, AttributeError):
                pass
            self.roi_rect_artist = None

        # Draw new ones if the image is present
        if self.image_on_canvas:
            if self.zero_pt:
                x, y = self.zero_pt
                self.zero_pt_artist, = self.ax.plot(x, y, 'r+', markersize=12, markeredgewidth=2)
            if self.roi_bbox:
                x_min, y_min, x_max, y_max = self.roi_bbox
                width = x_max - x_min
                height = y_max - y_min
                self.roi_rect_artist = self.ax.add_patch(
                    patches.Rectangle((x_min, y_min), width, height, fill=False, color='red', linewidth=2)
                )

    def get_zero_point(self):
        return self.zero_pt

    def get_roi_bbox(self):
        return self.roi_bbox

    def _setup_widgets(self):
        # --- Main Frames ---
        top_frame = ttk.Frame(self.window, padding="10")
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        controls_frame = ttk.Frame(self.window, padding="10")
        controls_frame.pack(side=tk.TOP, fill=tk.X)

        plot_frame = ttk.Frame(self.window)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Controls in Top Frame ---
        # Folder Path
        ttk.Label(top_frame, text="Source:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.path_var = tk.StringVar()
        path_entry = ttk.Entry(top_frame, textvariable=self.path_var, width=60)
        path_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        browse_folder_button = ttk.Button(top_frame, text="Select Folder...", command=self._browse_folder)
        browse_folder_button.grid(row=0, column=2, padx=2)
        browse_file_button = ttk.Button(top_frame, text="Select File...", command=self._browse_file)
        browse_file_button.grid(row=0, column=3, padx=2)

        # Sleep Time
        ttk.Label(top_frame, text="Sleep (s):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.sleep_var = tk.StringVar(value="2")
        ttk.Entry(top_frame, textvariable=self.sleep_var, width=5).grid(row=1, column=1, sticky=tk.W)

        # Start/Stop Buttons
        self.start_button = tk.Button(top_frame, text="Start Monitoring", command=self.start_monitoring, bg="green", fg="white", width=15)
        self.start_button.grid(row=0, column=4, padx=10)
        self.stop_button = tk.Button(top_frame, text="Stop Monitoring", command=self.stop_monitoring, bg="red", fg="white", width=15, state=tk.DISABLED)
        self.stop_button.grid(row=1, column=4, padx=10)
        
        top_frame.columnconfigure(1, weight=1)

        # --- New Controls Frame ---
        # Background Image
        ttk.Label(controls_frame, text="Background File:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.background_path_var = tk.StringVar()
        bg_path_entry = ttk.Entry(controls_frame, textvariable=self.background_path_var, width=60, state='readonly')
        bg_path_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))
        select_bg_button = ttk.Button(controls_frame, text="Select BG", command=self._select_background_image)
        select_bg_button.grid(row=0, column=2, padx=5)
        clear_bg_button = ttk.Button(controls_frame, text="Clear BG", command=self._clear_background_image)
        clear_bg_button.grid(row=0, column=3, padx=5)

        ttk.Label(controls_frame, text="BG Factor:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.bg_factor_var = tk.StringVar(value="1.0")
        ttk.Entry(controls_frame, textvariable=self.bg_factor_var, width=5).grid(row=0, column=5, sticky=tk.W)

        # Autosave Controls
        ttk.Label(controls_frame, text="Save Prefix:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.save_prefix_var = tk.StringVar()
        ttk.Entry(controls_frame, textvariable=self.save_prefix_var, width=20).grid(row=1, column=1, sticky=tk.W)
        
        self.autosave_var = tk.BooleanVar()
        autosave_check = ttk.Checkbutton(controls_frame, text="Autosave Image", variable=self.autosave_var)
        autosave_check.grid(row=1, column=2, padx=10)

        controls_frame.columnconfigure(1, weight=1)

        # --- Matplotlib Canvas in Plot Frame ---
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Awaiting Image...")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Bind analysis selection events to matplotlib event system
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_canvas_drag)
        self.canvas.mpl_connect('button_release_event', self.on_canvas_release)

        self.image_on_canvas = None
        self.colorbar = None

        # Toolbar
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # --- Colorbar Controls ---
        cbar_frame = ttk.Frame(toolbar_frame)
        cbar_frame.pack(side=tk.RIGHT, padx=10)
        ttk.Label(cbar_frame, text="VMin:").pack(side=tk.LEFT)
        self.vmin_var = tk.StringVar()
        ttk.Entry(cbar_frame, textvariable=self.vmin_var, width=7).pack(side=tk.LEFT)
        ttk.Label(cbar_frame, text="VMax:").pack(side=tk.LEFT)
        self.vmax_var = tk.StringVar()
        ttk.Entry(cbar_frame, textvariable=self.vmax_var, width=7).pack(side=tk.LEFT)
        update_cbar_button = ttk.Button(cbar_frame, text="Update", command=self.update_colormap)
        update_cbar_button.pack(side=tk.LEFT, padx=5)

    def _browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.path_var.set(folder_selected)
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def _browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select a single image file",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        if file_path:
            self.path_var.set(file_path)
            self.stop_monitoring() # Stop any active monitoring
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.DISABLED)
            self._load_single_image(file_path)

    def _load_single_image(self, file_path):
        try:
            img = Image.open(file_path)
            img_array = np.array(img)
            self.image_queue.put((img_array, os.path.basename(file_path)))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image file: {e}")

    def _select_background_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Background Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        if file_path:
            try:
                img = Image.open(file_path)
                self.background_image = np.array(img)
                self.background_path_var.set(file_path)
                messagebox.showinfo("Success", f"Background image '{os.path.basename(file_path)}' loaded.")
                # If an image is already displayed, re-process it with the new background
                if self.last_img_array is not None:
                    self.update_plot(self.last_img_array, self.last_filename)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load background image: {e}")
                self.background_image = None
                self.background_path_var.set("")

    def _clear_background_image(self):
        self.background_image = None
        self.background_path_var.set("")
        messagebox.showinfo("Info", "Background image cleared.")
        # If an image is already displayed, re-process it to remove the background
        if self.last_img_array is not None:
            self.update_plot(self.last_img_array, self.last_filename)

    def start_monitoring(self):
        if not self.path_var.get() or not os.path.isdir(self.path_var.get()):
            messagebox.showerror("Error", "Please select a valid folder path.")
            return
        
        self.stop_event.clear()
        
        # Initialize the set of seen files
        folder_path = self.path_var.get()
        try:
            self.seen_files = {
                os.path.join(folder_path, f) 
                for f in os.listdir(folder_path) 
                if os.path.isfile(os.path.join(folder_path, f))
            }
        except OSError as e:
            messagebox.showerror("Error", f"Could not read folder contents: {e}")
            return

        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.ax.set_title("Monitoring...")
        self.canvas.draw()

    def stop_monitoring(self):
        self.stop_event.set()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.ax.set_title("Monitoring Stopped.")
        self.canvas.draw()

    def _monitor_loop(self):
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        
        while not self.stop_event.is_set():
            try:
                folder_path = self.path_var.get()
                sleep_time = float(self.sleep_var.get())

                current_files = {
                    os.path.join(folder_path, f) 
                    for f in os.listdir(folder_path) 
                    if os.path.isfile(os.path.join(folder_path, f))
                }
                
                new_files = sorted(
                    list(current_files - self.seen_files), 
                    key=os.path.getmtime
                )

                for file_path in new_files:
                    # Check if the file is an image
                    if os.path.splitext(file_path)[1].lower() in image_extensions:
                        try:
                            img = Image.open(file_path)
                            img_array = np.array(img)
                            self.image_queue.put((img_array, os.path.basename(file_path)))
                        except Exception as e:
                            print(f"Error opening image {file_path}: {e}")
                
                self.seen_files = current_files
                time.sleep(sleep_time)

            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(sleep_time)

    def process_queue(self):
        try:
            img_array, filename = self.image_queue.get_nowait()
            self.update_plot(img_array, filename)
        except Empty:
            pass
        finally:
            self.window.after(100, self.process_queue)

    def update_plot(self, img_array, filename):
        self.last_img_array = img_array
        self.last_filename = filename
        # Trigger a full redraw and processing via update_colormap
        self.update_colormap()

    def _save_image(self, img_array, original_filename):
        # Determine the base path for saving
        source_path = self.path_var.get()
        if not source_path:
            print("Cannot save: Source path not set.")
            return

        prefix = self.save_prefix_var.get()
        if not prefix:
            prefix = "test_" # Default prefix if none is provided in GUI

        if os.path.isfile(source_path):
            # If source is a file, save next to it with prefix
            save_dir = os.path.dirname(source_path)
            base_name = os.path.basename(source_path)
            name, ext = os.path.splitext(base_name)
            new_filename = f"{prefix}{name}{ext}"
            save_path = os.path.join(save_dir, new_filename)
        elif os.path.isdir(source_path):
            # If source is a folder, create optapp_ folder as before
            parent_dir, monitor_basename = os.path.split(source_path)
            if not monitor_basename:
                monitor_basename = os.path.basename(source_path)
                parent_dir = os.path.dirname(source_path)

            save_folder_name = f"optapp_{monitor_basename}"
            save_dir = os.path.join(parent_dir, save_folder_name)
            os.makedirs(save_dir, exist_ok=True)

            new_filename = f"{prefix}{original_filename}"
            save_path = os.path.join(save_dir, new_filename)
        else:
            print(f"Cannot save: Invalid source path type: {source_path}")
            return

        try:
            img_to_save = Image.fromarray(img_array)
            img_to_save.save(save_path)
            print(f"Image saved to {save_path}")
        except Exception as e:
            print(f"Error saving image to {save_path}: {e}")

    def update_colormap(self):
        if self.last_img_array is None:
            if self.path_var.get() and os.path.isfile(self.path_var.get()):
                self._load_single_image(self.path_var.get())
            return

        # --- Start of logic moved from update_plot ---
        processed_img = self.last_img_array.copy()
        
        # Subtract background if it exists
        if self.background_image is not None:
            try:
                bg_factor = float(self.bg_factor_var.get())
                # Promote to a signed type to prevent underflow during subtraction
                original_dtype = processed_img.dtype
                promoted_img = processed_img.astype(np.int32)
                promoted_bg = (self.background_image * bg_factor).astype(np.int32)

                subtracted_img = np.subtract(promoted_img, promoted_bg)

                min_val = 0
                max_val = np.iinfo(original_dtype).max if np.issubdtype(original_dtype, np.integer) else 1.0
                
                clipped_img = np.clip(subtracted_img, min_val, max_val)
                processed_img = clipped_img.astype(original_dtype)
            except ValueError as e:
                messagebox.showwarning("Background Subtraction Error", f"Could not subtract background. Ensure dimensions match and factor is valid. Error: {e}")
            except Exception as e:
                messagebox.showwarning("Background Subtraction Error", f"An unexpected error occurred during background subtraction: {e}")

        if self.colorbar:
            self.colorbar.remove()
            self.colorbar = None

        self.ax.clear()
        self.image_on_canvas = self.ax.imshow(processed_img, cmap='viridis', aspect='equal')
        self.ax.set_title(self.last_filename)
        
        self.colorbar = self.fig.colorbar(self.image_on_canvas, ax=self.ax)
        # --- End of logic moved from update_plot ---

        try:
            vmin = self.vmin_var.get()
            vmax = self.vmax_var.get()
            
            new_vmin = float(vmin) if vmin else None
            new_vmax = float(vmax) if vmax else None
            
            self.image_on_canvas.set_clim(vmin=new_vmin, vmax=new_vmax)
        except ValueError:
            messagebox.showwarning("Input Error", "VMin/VMax must be numbers.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update colormap: {e}")

        self.draw_overlays()
        self.canvas.draw()

        # Autosave if enabled
        if self.autosave_var.get():
            self._save_image(processed_img, self.last_filename)

    def load_state(self):
        default_state = {
            "folder_path": "",
            "sleep_time": "2",
            "background_path": "",
            "save_prefix": "",
            "autosave": False,
            "vmin": "",
            "vmax": "",
            "zero_pt": None,
            "roi_bbox": None,
            "bg_factor": "1.0"
        }
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
            else:
                state = default_state
        except (IOError, json.JSONDecodeError):
            state = default_state

        self.path_var.set(state.get("folder_path", default_state["folder_path"]))
        self.sleep_var.set(state.get("sleep_time", default_state["sleep_time"]))
        self.background_path_var.set(state.get("background_path", default_state["background_path"]))
        self.save_prefix_var.set(state.get("save_prefix", default_state["save_prefix"]))
        self.autosave_var.set(state.get("autosave", default_state["autosave"]))
        self.vmin_var.set(state.get("vmin", default_state["vmin"]))
        self.vmax_var.set(state.get("vmax", default_state["vmax"]))
        self.zero_pt = state.get("zero_pt", default_state["zero_pt"])
        self.roi_bbox = state.get("roi_bbox", default_state["roi_bbox"])
        self.bg_factor_var.set(state.get("bg_factor", default_state["bg_factor"]))

        # Load the background image if path exists
        bg_path = self.background_path_var.get()
        if bg_path and os.path.exists(bg_path):
            try:
                self.background_image = np.array(Image.open(bg_path))
            except Exception as e:
                print(f"Failed to auto-load background image from state: {e}")
                self.background_path_var.set("")

    def save_state(self):
        state = {
            "folder_path": self.path_var.get(),
            "sleep_time": self.sleep_var.get(),
            "background_path": self.background_path_var.get(),
            "save_prefix": self.save_prefix_var.get(),
            "autosave": self.autosave_var.get(),
            "vmin": self.vmin_var.get(),
            "vmax": self.vmax_var.get(),
            "zero_pt": self.zero_pt,
            "roi_bbox": self.roi_bbox,
            "bg_factor": self.bg_factor_var.get()
        }
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=4)
        except IOError as e:
            print(f"Error saving state: {e}")

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
    
    def open_monitor():
        monitor_app = MonitorWindow(root)

    main_button = ttk.Button(root, text="Open Monitor", command=open_monitor)
    main_button.pack(pady=50, padx=50)
    
    root.mainloop()