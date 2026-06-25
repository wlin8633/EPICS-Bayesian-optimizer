# %% Imports
import time
from ophyd.signal import EpicsSignalRO, EpicsSignal
from ophyd import EpicsMotor, Component

# %%

def run_prerequisites():
    PVs = {
        "Power": "Station03:Relay1:OnOff",
    }

    print("Connecting to EPICS PVs...")
    for PV_name, PV_val in PVs.items():
        signal = EpicsSignal(PV_val, name=PV_name)
        try:
            signal.wait_for_connection()
            globals()[PV_name] = signal
            print(f"Connected to PV: {PV_name} ({PV_val})")
        except TimeoutError:
            print(f"Failed to connect to PV: {PV_name} ({PV_val})")

    # if power is off, turn it on
    if Power.get() == 0:
        Power.put(1)
        time.sleep(10)
    return
