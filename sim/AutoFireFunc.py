# %% # ----> Import packages
import json
import numpy as np
import matplotlib.pyplot as plt

import time
from ophyd.signal import EpicsSignalRO, EpicsSignal
from ophyd import EpicsMotor, Component

import gspread
from google.oauth2.service_account import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

def read_PVs(PVs_path) -> dict:
    with open(PVs_path, "r") as f:
        PVs = json.load(f)
    return PVs
        
def append_params(worksheet, param: dict):
    headers = worksheet.row_values(1)
    if headers == []:
        print("Worksheet is empty. Creating header row...")
        headers = list(param.keys())
        worksheet.append_row(headers)

    if headers != list(param.keys()):
        print("Error: Header mismatch between worksheet and param keys. Exiting...")
        return
    else:
        print("Header match confirmed. Appending data...")
        values = [value.get() if isinstance(value, EpicsSignal) else value for value in list(param.values())]
        worksheet.append_row(values)
        return

def authorize_gspread(creds_path, token_path) -> gspread.Client:
    # Define scopes
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    # Run OAuth flow
    flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
    creds = flow.run_local_server(port=0)
    # Save token for reuse
    with open(token_path, "w") as token:
        token.write(creds.to_json())
    # Authorize gspread
    client = gspread.authorize(creds)
    return client

def access_worksheet(worksheet_name, spreadsheet) -> gspread.Worksheet:
    worksheet = spreadsheet.worksheets()
    title_list = [ws.title for ws in worksheet]
    if worksheet_name in title_list:
        print(f"Worksheet {worksheet_name} found. Accessing...")
        return spreadsheet.worksheet(worksheet_name)
    else:
        print(f"Worksheet {worksheet_name} not found. Creating new one...")
        return spreadsheet.add_worksheet(worksheet_name, rows="100", cols="26")

def update_sheet_with_analysis(worksheet, shot_number_col, shot_number, analysis_results):
    """
    Finds a row by its shot number and updates it with analysis results.

    Args:
        worksheet (gspread.Worksheet): The worksheet object.
        shot_number_col (str): The name of the column containing the shot number.
        shot_number (int or str): The shot number to find.
        analysis_results (dict): A dictionary where keys are column names and
                                 values are the results to be updated.
    """
    try:
        # Get all headers to find column indices
        headers = worksheet.row_values(1)
        if not headers:
            print("Warning: Worksheet has no headers. Cannot update.")
            return

        # Find the column index for the shot number (1-based)
        try:
            shot_col_index = headers.index(shot_number_col) + 1
        except ValueError:
            raise ValueError(f"Shot number column '{shot_number_col}' not found in sheet headers.")

        # Find the cell with the matching shot number
        try:
            cell = worksheet.find(str(shot_number), in_column=shot_col_index)
        except gspread.CellNotFound:
            raise ValueError(f"Shot number '{shot_number}' not found in column '{shot_number_col}'.")

        # The row to update
        row_to_update = cell.row

        # Create a list of cells to update
        cells_to_update = []
        for col_name, value in analysis_results.items():
            try:
                # Find the column index for the result (1-based)
                result_col_index = headers.index(col_name) + 1
                cell_to_update = gspread.Cell(row=row_to_update, col=result_col_index, value=str(value))
                cells_to_update.append(cell_to_update)
            except ValueError:
                print(f"Warning: Result column '{col_name}' not found in sheet headers. Skipping.")
        
        # Update all cells in one batch
        if cells_to_update:
            worksheet.update_cells(cells_to_update)
            print(f"Successfully updated row {row_to_update} for shot {shot_number}.")

    except Exception as e:
        print(f"An error occurred while updating the sheet: {e}")
        # Re-raise the exception so the GUI can catch it
        raise e
