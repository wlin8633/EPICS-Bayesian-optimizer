# Bayesian Optimization Suite - User Manual

This manual provides a step-by-step guide on how to use the graphical user interfaces (GUIs) for experiment automation and data analysis.

## Table of Contents
1.  [**Main Workflow Overview**](#main-workflow-overview)
2.  [**Part 1: The Image Monitor (`ImageMonitor.py`)**](#part-1-the-image-monitor-imagemonitorpy)
    -   [Setting Up the Monitor](#setting-up-the-monitor)
    -   [Defining the Analysis Region (Crucial Step)](#defining-the-analysis-region-crucial-step)
3.  [**Part 2: The TP Analyzer (`TPAnalyzer.py`)**](#part-2-the-tp-analyzer-tpanalyzerpy)
    -   [Configuration](#configuration)
    -   [Standard Workflow (Manual Analysis)](#standard-workflow-manual-analysis)
    -   [Automated Workflow (Auto-Analysis Loop)](#automated-workflow-auto-analysis-loop)
4.  [**Part 3: The Main Optimizer (`OptimizerGUI.py`)**](#part-3-the-main-optimizer-optimizerguipy)
    -   [Connecting to Google Sheets](#connecting-to-google-sheets)
    -   [Defining the Optimization Problem](#defining-the-optimization-problem)
    -   [Running the Optimization](#running-the-optimization)
    -   [Setting Up the Automated Closed Loop](#setting-up-the-automated-closed-loop)

---

## Main Workflow Overview

The suite is designed for a closed-loop optimization workflow:

1.  **Suggest:** The `OptimizerGUI` suggests new machine parameters.
2.  **Apply:** You (or the script automatically) apply these parameters to the machine via EPICS.
3.  **Acquire:** The experiment runs, and a new image (or set of images) is saved.
4.  **Analyze:** The `TPAnalyzer` automatically processes the new image(s) and calculates a scalar objective value (e.g., total energy).
5.  **Record:** The `OptimizerGUI` automatically uploads this result to a Google Sheet.
6.  **Repeat:** The `OptimizerGUI` reads the updated history and suggests the next point.

This manual explains how to use the GUIs that manage this process.

---

## Part 1: The Image Monitor (`ImageMonitor.py`)

**Purpose:** To view live images from the experiment and, most importantly, to define the **Region of Interest (ROI)** and **Zero Point** for the Thomson Parabola analysis.

### Setting Up the Monitor

1.  **Open the Monitor:** Launch the `OptimizerGUI` and click the **"Open Image Monitor"** button.
2.  **Select Source:**
    -   To watch a folder for new images, click **"Select Folder..."** and choose the directory where your camera saves images.
    -   To view a single, static image, click **"Select File..."**.
3.  **Start Monitoring:** If you selected a folder, click the **"Start Monitoring"** button. The latest image will appear in the window.
4.  **Background Subtraction (Optional):**
    -   Click **"Select BG"** to choose a background image.
    -   This background will be subtracted from all subsequent images.
    -   Adjust the **"BG Factor"** if your background needs to be scaled (e.g., if it was taken with a different exposure time).

### Defining the Analysis Region (Crucial Step)

This is the most important function of the Image Monitor. The ROI and Zero Point you define here are used by the `TPAnalyzer`.

1.  **Enter Edit Mode:** With an image displayed, press **`Ctrl+E`** on your keyboard. The window title will change to "EDIT MODE".
2.  **Set the Zero Point:**
    -   The "Zero Point" is the undeflected spot on your detector (i.e., where particles would go if there were no fields).
    -   **Left-click** directly on this spot in the image. A red cross `+` will appear.
3.  **Set the Region of Interest (ROI):**
    -   The ROI is the rectangular area where your signal trace appears.
    -   **Click and drag** your mouse to draw a box around the entire Thomson parabola trace. A red rectangle will appear.
4.  **Exit Edit Mode:** Press **`Ctrl+E`** again to exit edit mode.
5.  **Done:** The Zero Point and ROI are now saved and can be accessed by the `TPAnalyzer`. You only need to do this once per experimental session, unless your alignment changes.

---

## Part 2: The TP Analyzer (`TPAnalyzer.py`)

**Purpose:** To perform the physics analysis on the TP images, calculate an energy spectrum, and save the results for the optimizer.

### Configuration

1.  **Open the Analyzer:** From the `OptimizerGUI`, click the **"TP Analyzer"** button.
2.  **Set Source Path:** Choose the folder where the raw images are saved. This is typically the same folder the `ImageMonitor` is watching.
3.  **Set Accumulation:** Enter the number of images to average for a single analysis (`Images to Accumulate`). This is useful for reducing shot-to-shot noise.
4.  **Analysis Parameters:** Fill in the physics parameters for your spectrometer (magnetic field `Bz`, drift distances `dB`, particle `Z` and `A`, etc.). These are critical for an accurate energy calculation.
5.  **Save Options:**
    -   Ensure **"Save Analysis Data"** is checked.
    -   Choose a **"Save Folder"** where the output `.npz` files will be stored. **This folder is what the `OptimizerGUI` will monitor in the final closed-loop setup.**
    -   Set a **"Save Prefix"** (e.g., `run1_`).

### Standard Workflow (Manual Analysis)

Use this workflow to analyze a single batch of images.

1.  **Update from Monitor:** First, ensure you have set the Zero Point and ROI in the `ImageMonitor`. Then, in the `TPAnalyzer` window, click **"Update ROI/Zero"**.
2.  **Start Accumulation:** Click **"Start Accumulation"**. The analyzer will wait for the specified number of new images to appear in the source folder.
3.  **Preview:** Once the images are collected, a preview will be generated. This shows the raw image with the analysis mask (the region around the theoretical parabola) overlaid.
    -   **Check the preview carefully!** If the mask does not align with the signal trace, adjust your analysis parameters (e.g., `Bz`, `zero_width`) or the Zero Point in the `ImageMonitor` and re-run the preview.
4.  **Run Analysis:** If the preview looks correct, click **"Confirm & Run Analysis"**.
    -   This performs the final calculation and saves the result as a `.npz` file in your designated save folder. A window will pop up showing the final calculated energy spectrum.

### Automated Workflow (Auto-Analysis Loop)

Use this for continuous, unattended data processing.

1.  **Enable the Loop:** Check the **"Auto Analysis Loop"** box.
2.  **Click "Start/Preview"**. The button text doesn't change, but it will initiate the loop.
3.  **Operation:** The analyzer will now continuously monitor the source folder.
    -   Every time it collects a full batch of new images (the number you set in `Images to Accumulate`), it will **automatically** run the full analysis pipeline (steps 3 and 4 from the manual workflow).
    -   It will save the `.npz` file and then immediately start waiting for the next batch of images.
4.  **To Stop:** Click the **"Stop"** button.

---

## Part 3: The Main Optimizer (`OptimizerGUI.py`)

**Purpose:** To manage the entire optimization process, from reading data and suggesting parameters to controlling the hardware.

### Connecting to Google Sheets

1.  **`Credentials Path`**: Path to your `credentials.json` file from the Google Cloud Console.
2.  **`Token Path`**: Path where the script can save the `token.json` file after you authorize it the first time.
3.  **`GSpread URL`**: The full URL of the Google Sheet you want to use.
4.  **`Sheet Name`**: The name of the specific worksheet (tab) within that spreadsheet.
5.  **`Shot Number Column`**: The exact name of the column that contains the unique shot number. This is used for updating rows with new analysis results.

### Defining the Optimization Problem

1.  **`Input Cols`**: A comma-separated list of the column names in your sheet that represent the machine parameters you are tuning (e.g., `"TargetX", "TargetY"`).
2.  **`Input Bounds`**: The allowed range for each input parameter, written as a comma-separated list of tuples. The order must match the `Input Cols`. Example: `(10.5, 11.5), (13.0, 14.0)`.
3.  **`Output Cols`**: A comma-separated list of the column names that represent the results from your analysis (e.g., `sum_spe_ene`).
4.  **`Objective`**:
    -   **Name:** The name for the final objective value that the script will calculate and try to optimize (e.g., `obj`).
    -   **Type:** `max` if you want to maximize the objective, `min` if you want to minimize it.
    -   **Formula:** A Python expression to calculate the objective. The output values are available in a numpy array `y`. Example: `y[0]` (if you only have one output), or `y[0] * y[1]` (to combine two outputs).

### Running the Optimization

1.  **Select Phase:**
    -   `Random`: Explores the parameter space randomly. Good for starting out.
    -   `Bayes`: The main optimization mode. Uses the model to make intelligent suggestions.
    -   `Local`: Fine-tunes the search around the best point found so far.
2.  **Get Suggestion:** Click **"Update and Suggest"**. The GUI will:
    -   Fetch the latest data from the Google Sheet.
    -   Calculate the objective for all historical points.
    -   Run the optimizer.
    -   Display the **"Next Suggested Params"** and the **"Current Best"** found so far.
3.  **Apply and Record:**
    -   Click **"Append Suggestion to Sheet"**. A confirmation box will appear, showing you which EPICS PVs will be updated.
    -   If you confirm, the script will:
        1.  Update the EPICS PVs on the machine with the new suggested values.
        2.  Append a new row to the Google Sheet with these parameter values, ready for the analysis result to be filled in later.

### Setting Up the Automated Closed Loop

This connects all the pieces together for fully automated experiments.

1.  **Configure `TPAnalyzer`:** Set up the `TPAnalyzer` in **Auto-Analysis Loop** mode as described above. Make sure its "Save Folder" is correctly specified.
2.  **Configure `OptimizerGUI`:**
    -   **`Analysis Folder to Monitor`**: Set this to the **exact same "Save Folder"** that `TPAnalyzer` is saving its `.npz` files into.
    -   **`Output Extraction Func`**: This should be `exp_output_extraction_function.py`. It tells the GUI how to open the `.npz` file and get the result.
    -   **`File Pattern`**: Set this to match the output files from the analyzer (e.g., `*_tp_analysis.npz`).
3.  **Start the Loop:**
    -   In `OptimizerGUI`, check the **"Auto-Optimization Loop"** box.
    -   Click **"Start Auto-Upload"**.
4.  **Operation:** The system is now live.
    -   The `OptimizerGUI` will wait for a new `.npz` file to appear in the analysis folder.
    -   When `TPAnalyzer` saves a new file, the `OptimizerGUI` detects it.
    -   It runs the `exp_output_extraction_function.py` to get the objective value.
    -   It uploads the result to the correct row in the Google Sheet (matching by shot number).
    -   It immediately runs the optimizer to get the *next* suggestion.
    -   It automatically updates the EPICS PVs and appends the new parameters to the sheet.
    -   The cycle repeats.
5.  **To Stop:** Click **"Stop Auto-Upload"** in the `OptimizerGUI` and **"Stop"** in the `TPAnalyzer`.
