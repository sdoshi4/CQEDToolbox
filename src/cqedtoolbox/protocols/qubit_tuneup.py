import torch 
from pathlib import Path

from instrumentserver import Client
from labcore.protocols import select_platform
from labcore.protocols.base import ProtocolBase, BranchBase
# from cqedtoolbox.protocols.operations import ResonatorSpectroscopy, ResonatorSpectroscopyVsGain, PiSpectroscopy, PowerRabi, ResonatorSpectroscopyAfterPi, ReadoutCalibration, SaturationSpectroscopy, T1Operation, T2EOperation, T2ROperation
from operations.single_qubit.res_spec import ResonatorSpectroscopy
from operations.single_qubit.res_spec_vs_gain import ResonatorSpectroscopyVsGain
from operations.single_qubit.sat_spec import SaturationSpectroscopy
from operations.fluxonium.res_spec_vs_flux import ResonatorSpectroscopyVsFlux
from operations.fluxonium.flux_offset_inference import FluxOffsetInference
from operations.fluxonium.sat_spec_vs_flux import SaturationSpectroscopyVsFlux, load_saved_resonator_curve, load_saved_qubit_curve
from operations.fluxonium.fluxonium_qubit_fit import FluxoniumQubitTheoryFit

import cqedtoolbox.instruments.qick.qick_sweep_v2 as qick_sweep_v2
from cqedtoolbox.protocols.configs.qick_config import QickConfig
from cqedtoolbox.instruments.qcodes_drivers.DMT.DMTCurrentSource import DMTCurrentSource
from cqedtoolbox import setup_measurements

from qcodes_contrib_drivers.drivers.QDevil.QDAC1 import QDac, Mode


instruments = Client(timeout = 30)
params = instruments.find_or_create_instrument('parameter_manager')


select_platform("QICK")

class QubitTuneup(ProtocolBase):

    def __init__(self, params, set_flux_current=None, cnn_checkpoint_dir=None,
                 report_path: Path = Path(".")):
        super().__init__(report_path)

        # The two analysis operations consume what the flux sweep produced, so
        # they are handed the instance rather than re-reading it from disk.
        res_spec_vs_flux = ResonatorSpectroscopyVsFlux(
            params, set_flux_current=set_flux_current
        )
        flux_offset = FluxOffsetInference(params, source=res_spec_vs_flux,
                                          checkpoint_dir=cnn_checkpoint_dir)

        # flux_offset = load_saved_resonator_curve(path=Path(r"Z:\Fluxonium\shaan_fluxonium_practice\Automation\CQEDToolbox\src\cqedtoolbox\data\2026-10-06\2026-10-06T113554_00131094-ResonatorSpectroscopyVsFlux"), analysis_name="ResonatorSpectroscopyVsFlux")
        # half_sweep = load_saved_qubit_curve(path=Path(r"Z:\Fluxonium\shaan_fluxonium_practice\Automation\CQEDToolbox\src\cqedtoolbox\analysis\SaturationSpectroscopyVsFlux"), analysis_name="2026-09-03T095822_e81567d7-SaturationSpectroscopyVsFlux", window="half", zero_current=params.q01.flux.zero_current(), half_current=params.q01.flux.half_current())

        half_sweep = SaturationSpectroscopyVsFlux(
            params, window="half", set_flux_current=set_flux_current,
            source=flux_offset)
        zero_sweep = SaturationSpectroscopyVsFlux(
            params, window="zero", set_flux_current=set_flux_current,
            source=flux_offset)

        self.root_branch = BranchBase("QubitTuneup")
        self.root_branch.extend([
            ResonatorSpectroscopy(params),
            ResonatorSpectroscopyVsGain(params),
            res_spec_vs_flux,
            flux_offset,
            half_sweep,
            FluxoniumQubitTheoryFit(params, sources=[half_sweep], label="half"),
            # zero_sweep,
            # FluxoniumQubitTheoryFit(params, sources=[half_sweep, zero_sweep],
            #                         label="final"),

            # SaturationSpectroscopy(params),


            # PowerRabi(params),
            # PiSpectroscopy(params),
            # ResonatorSpectroscopyAfterPi(params),
            # T1Operation(params),
            # T2ROperation(params),
            # T2EOperation(params),
            # ReadoutCalibration(params),
        ])

conf = QickConfig(params=params, nameserver_host="192.168.2.99", nameserver_name="myqick")
conf.generate_soccfg()
conf.soc.rfb_set_dac_filter(0, fc=6.45, ftype='bandpass', bw=1.0) # Readout resonator probe tone is 6.46GHz
conf.soc.rfb_set_dac_filter(1, fc=6, ftype='lowpass') # Qubit pulse varies in freq, but is less than 5.5GHz
conf.soc.rfb_set_adc_filter(4, fc=6.45, ftype='bandpass', bw=1.0) # Same as resonator probe tone

# This is by Channel
print("set DAC attenuators:", conf.soc.rfb_set_gen_rf(0, 10, 10)) # Resonator gets 20dB attenuation, qubit gets none
print("set DAC attenuators:", conf.soc.rfb_set_gen_rf(2, 10, 10)) # qubit gets none - I ADDED 10dB HERE
# ADC gets 20dB attenuation
print("set ADC attenuators:", conf.soc.rfb_set_ro_rf(0, 0))
conf.soc.rfb_set_rfadc_attenuator(4, 20) # By Port

qick_sweep_v2.config = conf

# these need to be specified so all measurement code is configured correctly
setup_measurements.options.instrument_clients = {'instruments': instruments}
setup_measurements.options.parameters = params

# currS = DMTCurrentSource("currSour", address="ASRL3::INSTR")
# FLUX_CHANNEL = "ch2"
# FLUX_RAMP_STEP = 0.25   # uA per step, walked rather than jumped
# FLUX_RAMP_DELAY = 0.001  # s between steps

# def set_flux_current(value):
#     """Ramp the flux bias to `value` (uA)."""
#     getattr(currS, FLUX_CHANNEL).ramp_current(
#         float(value), FLUX_RAMP_STEP, FLUX_RAMP_DELAY
#     )

qdac = QDac(name="qdac", address="ASRL4::INSTR")


R = 8.2*1e3 # 8.2kOhm
CHANNEL = "ch01"

def set_flux_current(value):
    voltage = value * 1e-6 * R
    if (voltage >= 10):
        raise ValueError("ERROR: Too high a voltage")
    getattr(qdac, CHANNEL).v(voltage)
    print("Set flux current to {} uA ({} V)".format(value, voltage))


# Directory holding the trained ensemble (flux_cnn_seed*.pt).  FluxOffsetInference
# loads every member: the spread of their parity probabilities is the confidence
# that decides whether the zero/half assignment is trustworthy.  The ensemble is
# specific to this device and this readout window, so it must be retrained
# (02_simulate_and_train.ipynb in Fluxonium-offset-inverse-model) for a new one.
CNN_CHECKPOINT_DIR = Path(
    "./analysis/flux_offset"
)

# currS.ch2.ramp_current(270, .5, .001)
set_flux_current(0)
QubitTuneup(params,
            set_flux_current=set_flux_current,
            cnn_checkpoint_dir=CNN_CHECKPOINT_DIR).execute()