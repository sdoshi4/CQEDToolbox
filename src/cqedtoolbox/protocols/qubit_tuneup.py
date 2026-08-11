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
from operations.fluxonium.fluxonium_theory_fit import FluxoniumResonatorTheoryFit

import cqedtoolbox.instruments.qick.qick_sweep_v2 as qick_sweep_v2
from cqedtoolbox.protocols.configs.qick_config import QickConfig
from cqedtoolbox.instruments.qcodes_drivers.DMT.DMTCurrentSource import DMTCurrentSource
from cqedtoolbox import setup_measurements


instruments = Client(timeout = 30)
params = instruments.find_or_create_instrument('parameter_manager')


select_platform("QICK")

class QubitTuneup(ProtocolBase):

    def __init__(self, params, set_flux_current=None, cnn_checkpoint=None,
                 report_path: Path = Path(".")):
        super().__init__(report_path)

        # The two analysis operations consume what the flux sweep produced, so
        # they are handed the instance rather than re-reading it from disk.
        res_spec_vs_flux = ResonatorSpectroscopyVsFlux(
            params, set_flux_current=set_flux_current
        )

        self.root_branch = BranchBase("QubitTuneup")
        self.root_branch.extend([
            # ResonatorSpectroscopy(params),
            # ResonatorSpectroscopyVsGain(params),
            res_spec_vs_flux,
            FluxOffsetInference(params, source=res_spec_vs_flux,
                                checkpoint_path=cnn_checkpoint),
            FluxoniumResonatorTheoryFit(params, source=res_spec_vs_flux),
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
conf.soc.rfb_set_dac_filter(1, fc=4, ftype='bandpass', bw = 2) # Qubit pulse varies in freq, but is less than 5.5GHz
conf.soc.rfb_set_adc_filter(4, fc=6.45, ftype='bandpass', bw=1.0) # Same as resonator probe tone

# This is by Channel
print("set DAC attenuators:", conf.soc.rfb_set_gen_rf(0, 10, 10)) # Resonator gets 20dB attenuation, qubit gets none
print("set DAC attenuators:", conf.soc.rfb_set_gen_rf(2, 10, 20)) # qubit gets none - I ADDED 10dB HERE
# ADC gets 20dB attenuation
print("set ADC attenuators:", conf.soc.rfb_set_ro_rf(0, 0))
conf.soc.rfb_set_rfadc_attenuator(4, 20) # By Port

qick_sweep_v2.config = conf

# these need to be specified so all measurement code is configured correctly
setup_measurements.options.instrument_clients = {'instruments': instruments}
setup_measurements.options.parameters = params

currS = DMTCurrentSource("currSour", address="ASRL3::INSTR")
FLUX_CHANNEL = "ch2"
FLUX_RAMP_STEP = 0.25   # uA per step, walked rather than jumped
FLUX_RAMP_DELAY = 0.001  # s between steps

def set_flux_current(value):
    """Ramp the flux bias to `value` (uA)."""
    getattr(currS, FLUX_CHANNEL).ramp_current(
        float(value), FLUX_RAMP_STEP, FLUX_RAMP_DELAY
    )


# Trained offset-inverse CNN. Produced by the Fluxonium-offset-inverse-model
# repo; FluxOffsetInference fails cleanly if this is missing.
CNN_CHECKPOINT = "fluxonium_inverse_cnn_better.pt"


QubitTuneup(params,
            set_flux_current=set_flux_current,
            cnn_checkpoint=CNN_CHECKPOINT).execute()