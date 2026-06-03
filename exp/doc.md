# Experiment Automation and Optimization Suite Documentation

This document provides an overview of the Python scripts and configuration files for the Bayesian Optimization and experiment control suite.

## Table of Contents
1.  [Core Application Logic](#core-application-logic)
    -   [`OptimizerGUI.py`](#optimizerguipy)
    -   [`BayesianOptimization.py`](#bayesianoptimizationpy)
2.  [Diagnostic & Analysis Tools](#diagnostic--analysis-tools)
    -   [`ImageMonitor.py`](#imagemonitorpy)
    -   [`TPAnalyzer.py`](#tpanalyzerpy)
3.  [Helper Modules & Functions](#helper-modules--functions)
    -   [`AutoFireFunc.py`](#autofirefuncpy)
    -   [`exp_output_extraction_function.py`](#exp_output_extraction_functionpy)
    -   [`exp_prerequisite_function.py`](#exp_prerequisite_functionpy)
4.  [Configuration Files](#configuration-files)
    -   [`optimizer_params.json`](#optimizer_paramsjson)
    -   [`gui_state.json`](#gui_statejson)
    -   [`monitor_state.json`](#monitor_statejson)
    -   [`tp_analyzer_state.json`](#tp_analyzer_statejson)

---

## Core Application Logic

### `OptimizerGUI.py`

This is the main graphical user interface (GUI) for the optimization suite, built using Tkinter. It serves as the central control panel for configuring, running, and monitoring the Bayesian optimization process.

**Key Features:**
- **Google Sheet Integration:** Configures paths to credentials, Google Sheet URL, and specific sheet names to read historical data from and append new suggestions to.
- **Optimizer Configuration:** Allows the user to define input parameters (variables), their bounds, output parameters (objectives), and the formula to calculate the final objective.
- **Phase Control:** Users can select the optimization phase:
    - `random`: Randomly samples the parameter space.
    - `bayes`: Uses a Gaussian Process model to suggest the next point based on Expected Improvement (EI).
    - `local`: Performs a local search around the current best-known point.
- **Auto-Upload & Optimization Loop:** Can monitor a folder for new analysis files, automatically extract results, upload them to the Google Sheet, and trigger the next optimization cycle. This creates a closed-loop system for automated experiments.
- **EPICS Integration:** Reads `optimizer_params.json` to map suggested parameter values to EPICS Process Variables (PVs) and pushes updates to the control system.
- **Launches Other Tools:** Provides buttons to open the `ImageMonitor` and `TPAnalyzer` windows.

### `BayesianOptimization.py`

This script contains the core logic for the Bayesian optimization algorithm, leveraging the `scikit-optimize` library.

**Key Functions:**
- **`suggest_next_params_bo(...)`**: The main function that takes the historical data (as a pandas DataFrame), input parameter definitions, and the current optimization phase to suggest the next set of parameters to try.
- **`_clean_history(...)`**: Pre-processes the data from the Google Sheet to handle missing values and ensure correct data types.
- **`_objective_to_skopt_y(...)`**: Prepares the objective values for `scikit-optimize`, which always minimizes. If the goal is maximization, it negates the objective values.
- **Optimization Phases:**
    - **`_sample_uniform(...)`**: Generates a random point within the defined bounds.
    - **`_local_perturb(...)`**: Takes the best point found so far and suggests a new point in its immediate vicinity.
    - **Bayesian Phase**: Fits a Gaussian Process (GP) model to the existing data and uses the "Expected Improvement" (EI) acquisition function to propose the next point that is most likely to be a global optimum.

The script also includes a `if __name__ == "__main__":` block that demonstrates a simulated 3-phase optimization run on a synthetic function, which is useful for testing the algorithm's behavior.

---

## Diagnostic & Analysis Tools

### `ImageMonitor.py`

A Tkinter-based tool for live monitoring of image files from an experiment. It can watch a folder for new images or display a single static image.

**Key Features:**
- **Live Monitoring:** Watches a specified directory and displays the most recent image file.
- **Background Subtraction:** Allows loading a background image, which can be subtracted from the primary image. A scaling factor can be applied to the background.
- **ROI and Zero-Point Selection:** Provides an interactive "Edit Mode" (`Ctrl+E`) where the user can:
    - **Left-click:** Set a "zero point" (e.g., the undeflected beam spot).
    - **Click and drag:** Define a rectangular Region of Interest (ROI) for analysis.
- **Autosave:** Can automatically save the processed (e.g., background-subtracted) image with a user-defined prefix.
- **State Persistence:** Saves the UI state (paths, ROI, etc.) to `monitor_state.json`.

### `TPAnalyzer.py`

A tool for analyzing Thomson Parabola (TP) data. It takes raw image data (often from the `ImageMonitor`) and calculates an energy spectrum.

**Key Features:**
- **Image Accumulation:** Can process a single image or accumulate a specified number of images and perform analysis on their mean and standard deviation, which helps reduce noise.
- **Auto-Analysis Loop:** Can be configured to automatically watch a folder. When enough new images are detected, it processes them as a batch, performs the analysis, and saves the result.
- **Integration with ImageMonitor:** Pulls the critical **Zero Point** and **Analysis ROI** directly from the `ImageMonitor` window to define the analysis region and coordinate system.
- **Physics-Based Analysis:**
    1.  Calculates theoretical parabola traces based on spectrometer parameters (magnetic/electric fields, geometry).
    2.  Creates a mask around the theoretical trace.
    3.  Applies this mask to the ROI of the input image to isolate the proton signal.
    4.  Projects the masked signal onto the energy axis to generate a spectrum.
- **Data Export:** Saves the analysis results, including the calculated energy spectrum, configuration parameters, and the original shot number, into a `.npz` file (e.g., `sbgtest_379_tp_analysis.npz`). This file is then read by the `exp_output_extraction_function.py`.

---

## Helper Modules & Functions

### `AutoFireFunc.py`

A utility module containing helper functions for interacting with Google Sheets and, implicitly, for structuring data for EPICS updates.

**Key Functions:**
- **`authorize_gspread(...)`**: Handles the OAuth2 authentication flow with Google APIs to get authorization for accessing spreadsheets.
- **`access_worksheet(...)`**: Finds a worksheet by name within a spreadsheet, creating it if it doesn't exist.
- **`update_sheet_with_analysis(...)`**: A crucial function used by the auto-upload loop. It finds a row in the Google Sheet based on a `shot_number` and updates it with the analysis results (e.g., `sum_spe_ene`) extracted from a `.npz` file.

### `exp_output_extraction_function.py`

This is a **user-defined** script designed to be a plug-in for the `OptimizerGUI`. Its purpose is to bridge the gap between the output of an analysis tool (like `TPAnalyzer`) and the main optimization loop.

**Key Function:**
- **`process_analysis_file(file_path)`**:
    - Takes the path to a data file (e.g., a `.npz` file).
    - Loads the data.
    - **Extracts the `shot_number`**, which is essential for matching the data to the correct row in the Google Sheet.
    - **Performs calculations** to derive the final scalar objective value(s) from the raw analysis output. For example, it calculates `sum_spe_ene` by integrating the product of the energy axis and the spectrum.
    - Returns a dictionary where keys match the column names in the Google Sheet.

### `exp_prerequisite_function.py`

This is another **user-defined** script that can be executed before a new set of parameters is applied to the machine.

**Key Function:**
- **`run_prerequisites()`**: This function is called by the `OptimizerGUI` right before it sends EPICS updates. It is intended to contain checks or actions that must be performed to ensure the machine is in a safe state to receive new parameters. The example provided toggles the state of two EPICS PVs related to over-voltage and over-current protection.

---

## Configuration Files

These JSON files are used to store the state of the various GUI components, allowing users to close and re-open the application without losing their settings.

### `optimizer_params.json`

A critical configuration file that maps the human-readable names of the optimization input variables to their corresponding EPICS Process Variable (PV) strings.

- **Example:** `{"TargetX": "TP_Target:Motor_X.VAL", "TargetY": "TP_Target:Motor_Y.VAL"}`
- **Function:** When the optimizer suggests a new value for `TargetX`, the `OptimizerGUI` uses this file to know it must write that value to the `TP_Target:Motor_X.VAL` PV.

### `gui_state.json`

Saves the state of the main `OptimizerGUI` window. This includes:
- Paths to credential files.
- Google Sheet URL and name.
- Definitions of input/output columns, bounds, and objective formula.
- The currently selected optimization phase.
- Paths for auto-upload monitoring.

### `monitor_state.json`

Saves the state of the `ImageMonitor` window. This includes:
- The folder or file path being monitored.
- The path to the background image.
- The VMin/VMax values for the color map.
- The coordinates of the **Zero Point** and **ROI Bbox**.

### `tp_analyzer_state.json`

Saves the state of the `TPAnalyzer` window. This includes:
- The source path for images.
- The number of images to accumulate.
- All physics and analysis parameters (spectrometer geometry, particle properties, etc.).
- Save path and prefix for the output `.npz` files.
