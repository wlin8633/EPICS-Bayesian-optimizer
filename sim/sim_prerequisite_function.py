# %% Imports
import time
from ophyd.signal import EpicsSignalRO, EpicsSignal
from ophyd import EpicsMotor, Component

# %%

def run_prerequisites():
    PVs = {
        "p1": "Station03:GPP3323:Ch2:OVPState",
        "p2": "Station03:GPP3323:Ch2:OCPState",
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

    # if 0, put 1; if 1, put 0
    if p1.get() == 0:
        p1.put(1)
    elif p1.get() == 1:
        p1.put(0)
        
    if p2.get() == 0:
        p2.put(1)
    elif p2.get() == 1:
        p2.put(0)
        
    return
