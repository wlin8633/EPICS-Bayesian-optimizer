# %% Imports
import time
from ophyd.signal import EpicsSignalRO, EpicsSignal
from ophyd import EpicsMotor, Component
import random

# %%

def run_post_recording():
    PVs = {
        "p1": "Station03:GPP3323:Ch2:Vset",
        "p2": "Station03:GPP3323:Ch2:Iset",
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
    # if Power.get() == 0:
    #     Power.put(1)
    #     time.sleep(10)
    p1.put(random.uniform(0, 1))
    p2.put(random.uniform(0, 1))
    return
