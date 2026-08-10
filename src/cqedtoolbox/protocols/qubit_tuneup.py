from pathlib import Path

from instrumentserver import Client
from labcore.protocols import select_platform
from labcore.protocols.base import ProtocolBase, BranchBase
# from cqedtoolbox.protocols.operations import ResonatorSpectroscopy, ResonatorSpectroscopyVsGain, PiSpectroscopy, PowerRabi, ResonatorSpectroscopyAfterPi, ReadoutCalibration, SaturationSpectroscopy, T1Operation, T2EOperation, T2ROperation
from operations.single_qubit.res_spec import ResonatorSpectroscopy
from operations.single_qubit.res_spec_vs_gain import ResonatorSpectroscopyVsGain
from operations.single_qubit.sat_spec import SaturationSpectroscopy
from operations.fluxonium.res_spec_vs_flux import ResonatorSpectroscopyVsFlux

import cqedtoolbox.instruments.qick.qick_sweep_v2 as qick_sweep_v2
from cqedtoolbox.protocols.configs.qick_config import QickConfig
from cqedtoolbox import setup_measurements


instruments = Client(timeout = 30)
params = instruments.find_or_create_instrument('parameter_manager')


select_platform("QICK")

class QubitTuneup(ProtocolBase):

    def __init__(self, params, report_path: Path = Path(".")):
        super().__init__(report_path)

        self.root_branch = BranchBase("QubitTuneup")
        self.root_branch.extend([
            # ResonatorSpectroscopy(params),
            # ResonatorSpectroscopyVsGain(params),
            ResonatorSpectroscopyVsFlux(params)
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

# This is by Port

QubitTuneup(params).execute()