# %%
import numpy as np
import os
import re

def process_analysis_file(file_path):
    """
    Processes a single analysis data file (e.g., .npz) to extract scalar output parameters.

    This function is designed to be modified by the user to fit their specific analysis needs.
    The framework will automatically call this function when a new analysis file is detected.

    Args:
        file_path (str): The absolute path to the new analysis file.

    Returns:
        dict: A dictionary containing the results. The dictionary MUST include a 'shot_number' key.
              The other keys should be the names of the columns in your Google Sheet that you want 
              to update, and their values should be the calculated scalar parameters.
              
              Example:
              {
                  'shot_number': 15,
                  'y1': 500.7,          # Total counts in the spectrum
                  'y2': 12.5,           # Peak energy in MeV
                  'some_other_col': 0.95 # Another calculated value
              }
              
              If the file cannot be processed or is invalid, return None.
    """
    try:
        # --- 1. Load the data file ---
        # This example assumes a .npz file created by TPAnalyzer.py
        with np.load(file_path, allow_pickle=True) as data:
            
            # --- 2. Extract the shot number ---
            # The 'shot_number' key is essential for finding the correct row in the Google Sheet.
            if 'shot_number' in data:
                shot_number = data['shot_number'].item()
            else:
                # If no shot number is in the file, you might try to get it from the filename
                # as a fallback. This example extracts digits from the filename.
                filename = os.path.basename(file_path)
                shot_number_match = re.search(r'\d+', filename)
                if shot_number_match:
                    shot_number = int(shot_number_match.group(0))
                else:
                    print(f"Warning: Could not determine shot number for {file_path}")
                    return None # Cannot proceed without a shot number

            # --- 3. Perform your custom calculations ---
            # This is the section you should customize.
            # The example below calculates two metrics from a Thomson Parabola spectrum.

            # --- 4. Assemble the results dictionary ---
            # Automatically extract all scalar values from the npz file
            results = {"shot_number": shot_number}
            for key in data.files:
                if key != "shot_number":
                    val = data[key]
                    results[key] = val.item() if hasattr(val, 'item') else val
            
            return results

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# if __name__ == '__main__':
#     # %% read a .npz file 
#     import os
#     import numpy as np
#     import matplotlib.pyplot as plt
#     # %%
#     ds = np.load(r"C:\path\to\your\test_analysis.npz", allow_pickle=True)
#     ds.files

#     # %% plot the mean spectrum
#     mean_spectrum = ds['mean_spectrum']
#     energy_axis = ds['energy_axis']
#     plt.plot(energy_axis, mean_spectrum)

#     # %% find the peak counts
#     peak_index = np.argmax(mean_spectrum)
#     ds['shot_number'].item()
#     # %%

# %%
