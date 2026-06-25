# %%
import time
from caproto.server import pvproperty, PVGroup, ioc_arg_parser, run

class BO_Sim_IOC(PVGroup):
    """
    A simulated EPICS IOC for Bayesian Optimization testing.
    Provides 7 PVs corresponding to TNSA optimization parameters (x1 - x7).
    """
    # x1: Laser Energy (J)
    x1 = pvproperty(value=11.0, name='Station03:Sim:LaserE')
    # x2: Target Thickness (um)
    x2 = pvproperty(value=2.0, name='Station03:Sim:TargetThick')
    # x3: Prepulse Contrast Ratio
    x3 = pvproperty(value=0.8, name='Station03:Sim:Contrast')
    # x4: Defocus
    x4 = pvproperty(value=0.0, name='Station03:Sim:Defocus')
    # x5: Astigmatism
    x5 = pvproperty(value=150.0, name='Station03:Sim:Astig')
    # x6: Coma X
    x6 = pvproperty(value=0.005, name='Station03:Sim:ComaX')
    # x7: Coma Y
    x7 = pvproperty(value=100.0, name='Station03:Sim:ComaY')

if __name__ == '__main__':
    ioc_options, run_options = ioc_arg_parser(
        default_prefix='',
        desc="Simulated PVs for Bayesian Optimization test."
    )
    print("Starting Simulated EPICS IOC for 7 parameters...")
    print("PVs available:")
    print(" - Station03:Sim:LaserE        (x1)")
    print(" - Station03:Sim:TargetThick   (x2)")
    print(" - Station03:Sim:Contrast      (x3)")
    print(" - Station03:Sim:Defocus       (x4)")
    print(" - Station03:Sim:Astig         (x5)")
    print(" - Station03:Sim:ComaX         (x6)")
    print(" - Station03:Sim:ComaY         (x7)")
    print("\nWaiting for connections...")
    
    ioc = BO_Sim_IOC(**ioc_options)
    run(ioc.pvdb, **run_options)

# %%
