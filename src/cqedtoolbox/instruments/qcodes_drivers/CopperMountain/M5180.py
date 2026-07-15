# This Python file uses the following encoding: utf-8
# Etienne Dumur <etienne.dumur@gmail.com>, august 2020
# Simon Zihlmannr <zihlmann.simon@gmail.com>, february/march 2021
# Shaan Doshi <shaand3@illinois,edu>, july 2026

# Original qcodes driver: https://github.com/QCoDeS/Qcodes_contrib_drivers/blob/main/src/qcodes_contrib_drivers/drivers/CopperMountain/M5180.py
# Shaan Doshi: Changed to match Keysight/Keysight_P9374A.py driver to work with general vna.py functions (tested with sweeps but not yet with twotone qubit spec)

import logging
import numpy as np
import cmath, math
from typing import Tuple, Any, List
 
from qcodes.instrument import VisaInstrument, InstrumentChannel
from qcodes.validators import Numbers, Enum, Ints, Bool
from qcodes.parameters import create_on_off_val_mapping
 
from qcodes.parameters import (
    MultiParameter,
    ManualParameter,
    ParamRawDataType
)
 
log = logging.getLogger(__name__)
 
class Trace(InstrumentChannel):
    """
    Per-trace/measurement parameters, mirroring the shape of
    Keysight_P9374A_SingleChannel's Trace channel (npts, frequency, data,
    s_parameter) for cross-driver compatibility.
    """
 
    def __init__(self, parent: "M5180", number: int, name: str, **kwargs):
        self._number = number
        super().__init__(parent, name=name, **kwargs)
 
        self.add_parameter('npts',
                           unit='',
                           get_cmd=lambda: self.parent.num_points(),
                           set_cmd=self._reject_read_only_set,
                           snapshot_exclude=True,
                           docstring='number of points in the trace')
 
        self.add_parameter('frequency',
                           unit='Hz',
                           get_cmd=self._get_frequency,
                           set_cmd=self._reject_read_only_set,
                           snapshot_exclude=True)

        self.add_parameter('data',
                           unit='',
                           get_cmd=self._get_data,
                           set_cmd=self._reject_read_only_set,
                           snapshot_exclude=True)
 
        self.add_parameter('s_parameter',
                           get_cmd=self._get_s_parameter,
                           set_cmd=self._set_s_parameter,
                           vals=Enum('S11', 'S12', 'S21', 'S22'),
                           get_parser=str,
                           snapshot_exclude=True)
        
    def _reject_read_only_set(self, value: Any) -> None:
        """Keep trace acquisition parameters read-only.
 
        Older InstrumentServer GUIs call ``setValue`` on every broadcast
        parameter update, including read-only array parameters.  Giving these
        parameters a setter makes that GUI use a value-capable editor.  A real
        set remains forbidden and returns this clear error to the caller.
        """
        raise RuntimeError(f"{self.full_name} is read-only")
 
    def _get_frequency(self) -> np.ndarray:
        freq_raw = self.ask("SENS1:FREQ:DATA?")
        return np.fromstring(freq_raw, dtype=float, sep=',')
 
    # def _get_data(self) -> np.ndarray:
    #     raw = self.ask(f"CALC1:TRAC{self._number}:DATA:FDAT?")
    #     return np.fromstring(raw, dtype=float, sep=',')
    def _get_data(self) -> np.ndarray:
        prev_format = self.ask(f"CALC1:TRAC{self._number}:FORM?")
        self.write(f"CALC1:TRAC{self._number}:FORM POL")
        raw = self.ask(f"CALC1:TRAC{self._number}:DATA:FDAT?")
        self.write(f"CALC1:TRAC{self._number}:FORM {prev_format}")
        data = np.fromstring(raw, dtype=float, sep=',')

        # Validity check
        expected_size = 2 * self.parent.num_points()
        if data.size != expected_size:
            raise RuntimeError(
                f"Expected {expected_size} real/imaginary values from trace "
                f"{self._number}, but received {data.size}."
            )
        
        return data[0::2] + 1j*data[1::2]
 
    def _get_s_parameter(self) -> str:
        return self.ask(f"CALC1:PAR{self._number}:DEF?").strip().strip('"')
 
    def _set_s_parameter(self, s_param: str) -> None:
        self.write(f"CALC1:PAR{self._number}:DEF {s_param}")
 
class FrequencySweepMagPhase(MultiParameter):
    """
    Sweep that returns magnitude and phase.
    """
 
    def __init__(self,
        name: str,
        start: float,
        stop: float,
        npts: int,
        instrument: "M5180",
        **kwargs: Any,
        ) -> None:
        """
        Linear frequency sweep that returns magnitude and phase for a single
        trace.
 
        Args:
            name (str): Name of the linear frequency sweep
            start (float): Start frequency of linear sweep
            stop (float): Stop frequency of linear sweep
            npts (int): Number of points of linear sweep
            instrument: Instrument to which sweep is bound to.
        """
        super().__init__(
            name,
            instrument=instrument,
            names=(
                f"{instrument.short_name}_{name}_magnitude",
                f"{instrument.short_name}_{name}_phase"),
            labels=(
                f"{instrument.short_name} {name} magnitude",
                f"{instrument.short_name} {name} phase",
            ),
            units=("dB", "rad"),
            setpoint_units=(("Hz",), ("Hz",)),
            setpoint_labels=(
                (f"{instrument.short_name} frequency",),
                (f"{instrument.short_name} frequency",),
            ),
            setpoint_names=(
                (f"{instrument.short_name}_frequency",),
                (f"{instrument.short_name}_frequency",),
            ),
            shapes=((npts,), (npts,),),
            **kwargs,
        )
        self.set_sweep(start, stop, npts)
 
    def set_sweep(self, start: float, stop: float, npts: int) -> None:
        """Updates the setpoints and shapes based on start, stop and npts.
 
        Args:
            start (float): start frequency
            stop (float): stop frequency
            npts (int): number of points
        """
        f = tuple(np.linspace(int(start), int(stop), num=npts))
        self.setpoints = ((f,), (f,))
        self.shapes = ((npts,), (npts,))
 
    def get_raw(self) -> Tuple[ParamRawDataType, ParamRawDataType]:
        """Gets data from instrument
 
        Returns:
            Tuple[ParamRawDataType, ...]: magnitude, phase
        """
        assert isinstance(self.instrument, M5180)
        self.instrument.write('CALC1:PAR:COUN 1') # 1 trace
        self.instrument.write('CALC1:PAR1:DEF {}'.format(self.name))
        self.instrument.trigger_source('bus') # set the trigger to bus
        self.instrument.write('TRIG:SEQ:SING') # Trigger a single sweep
        self.instrument.ask('*OPC?') # Wait for measurement to complete
 
        # get data from instrument
        self.instrument.write('CALC1:TRAC1:FORM POL')  # ensure correct format
        sxx_raw = self.instrument.ask("CALC1:TRAC1:DATA:FDAT?")
        self.instrument.write('CALC1:TRAC1:FORM MLOG')
 
        # Get data as numpy array
        sxx = np.fromstring(sxx_raw, dtype=float, sep=',')
        sxx = sxx[0::2] + 1j*sxx[1::2]
 
        return self.instrument._db(sxx), np.unwrap(np.angle(sxx))
 
 
class PointMagPhase(MultiParameter):
    """
    Returns the average Sxx of a frequency sweep.
    Work around for a CW mode where only one point is read.
    npts=2 and stop = start + 1 (in Hz) is required.
    """
 
    def __init__(self,
        name: str,
        instrument: "M5180",
        **kwargs: Any,
        ) -> None:
        """Magnitude and phase measurement of a single point at start
        frequency.
 
        Args:
            name (str): Name of point measurement
            instrument:  Instrument to which parameter is bound to.
        """
 
        super().__init__(
            name,
            instrument=instrument,
            names=(
                f"{instrument.short_name}_{name}_magnitude",
                f"{instrument.short_name}_{name}_phase"),
            labels=(
                f"{instrument.short_name} {name} magnitude",
                f"{instrument.short_name} {name} phase",
            ),
            units=("dB", "rad"),
            setpoints=((), (),),
            shapes=((), (),),
            **kwargs,
        )
 
    def get_raw(self) -> Tuple[ParamRawDataType, ParamRawDataType]:
        """Gets data from instrument
 
        Returns:
            Tuple[ParamRawDataType, ...]: magnitude, phase
        """
 
        assert isinstance(self.instrument, M5180)
        # check that npts, start and stop fullfill requirements if point_check_sweep_first is True.
        if self.instrument.point_check_sweep_first():
            if self.instrument.num_points() != 2:
                raise ValueError('Npts is not 2 but {}. Please set it to 2'.format(self.instrument.num_points()))
            if self.instrument.fstop() - self.instrument.fstart() != 1:
                raise ValueError('Stop-start is not 1 Hz but {} Hz. Please adjust'
                                'start or stop.'.format(self.instrument.fstop()-self.instrument.fstart()))
 
        self.instrument.write('CALC1:PAR:COUN 1') # 1 trace
        self.instrument.write('CALC1:PAR1:DEF {}'.format(self.name[-3:]))
        self.instrument.trigger_source('bus') # set the trigger to bus
        self.instrument.write('TRIG:SEQ:SING') # Trigger a single sweep
        self.instrument.ask('*OPC?') # Wait for measurement to complete
 
        # get data from instrument
        self.instrument.write('CALC1:TRAC1:FORM POL')  # ensure correct format
        sxx_raw = self.instrument.ask("CALC1:TRAC1:DATA:FDAT?")
 
        # Get data as numpy array
        sxx = np.fromstring(sxx_raw, dtype=float, sep=',')
        sxx = sxx[0::2] + 1j*sxx[1::2]
 
        # Return the average of the trace, which will have "start" as
        # its setpoint
        sxx_mean = np.mean(sxx)
        return 20*math.log10(abs(sxx_mean)), (cmath.phase(sxx_mean))
 
 
 
class PointIQ(MultiParameter):
    """
    Returns the average Sxx of a frequency sweep, in terms of I and Q.
    Work around for a CW mode where only one point is read.
    npts=2 and stop = start + 1 (in Hz) is required.
    """
 
    def __init__(self,
        name: str,
        instrument: "M5180",
        **kwargs: Any,
        ) -> None:
        """I and Q measurement of a single point at start
        frequency.
 
        Args:
            name (str): Name of point measurement
            instrument:  Instrument to which parameter is bound to.
        """
 
        super().__init__(
            name,
            instrument=instrument,
            names=(
                f"{instrument.short_name}_{name}_i",
                f"{instrument.short_name}_{name}_q"),
            labels=(
                f"{instrument.short_name} {name} i",
                f"{instrument.short_name} {name} q",
            ),
            units=("V", "V"),
            setpoints=((), (),),
            shapes=((), (),),
            **kwargs,
        )
 
    def get_raw(self) -> Tuple[ParamRawDataType, ParamRawDataType]:
        """Gets data from instrument
 
        Returns:
            Tuple[ParamRawDataType, ...]: I, Q
        """
 
        assert isinstance(self.instrument, M5180)
        # check that npts, start and stop fullfill requirements if point_check_sweep_first is True.
        if self.instrument.point_check_sweep_first():
            if self.instrument.num_points() != 2:
                raise ValueError('Npts is not 2 but {}. Please set it to 2'.format(self.instrument.num_points()))
            if self.instrument.fstop() - self.instrument.fstart() != 1:
                raise ValueError('Stop-start is not 1 Hz but {} Hz. Please adjust'
                                'start or stop.'.format(self.instrument.fstop()-self.instrument.fstart()))
 
        self.instrument.write('CALC1:PAR:COUN 1') # 1 trace
        self.instrument.write('CALC1:PAR1:DEF {}'.format(self.name[-3:]))
        self.instrument.trigger_source('bus') # set the trigger to bus
        self.instrument.write('TRIG:SEQ:SING') # Trigger a single sweep
        self.instrument.ask('*OPC?') # Wait for measurement to complete
 
        # get data from instrument
        self.instrument.write('CALC1:TRAC1:FORM POL')  # ensure correct format
        sxx_raw = self.instrument.ask("CALC1:TRAC1:DATA:FDAT?")
 
        # Get data as numpy array
        sxx = np.fromstring(sxx_raw, dtype=float, sep=',')
 
        # Return the average of the trace, which will have "start" as
        # its setpoint
        return np.mean(sxx[0::2]), np.mean(sxx[1::2])
 
 
 
class M5180(VisaInstrument):
    """
    This is the QCoDeS python driver for the VNA M5180 from Copper Mountain
    """
 
    def __init__(self, name       : str,
                       address    : str,
                       terminator : str="\n",
                       timeout    : int=100000,
                       **kwargs):
        """
        QCoDeS driver for the VNA M5180 from Copper Mountain.
        This driver only uses one channel.
 
        Args:
            name (str): Name of the instrument.
            address (str): Address of the instrument.
            terminator (str): Terminator character of
                the string reply. Optional, default ``"\\n"``
            timeout (int): VISA timeout is set purposely
                to a long time to allow long spectrum measurement.
                Optional, default 100000
        """
 
        super().__init__(name       = name,
                         address    = address,
                         terminator = terminator,
                         timeout    = timeout,
                         **kwargs)
 
        self.add_function('reset', call_cmd='*RST')
 
        # set the unit of the electrical distance to meter
        self.write('CALC1:CORR:EDEL:DIST:UNIT MET')
 
        self.add_parameter(name='rfout',
                           label='Output',
                           get_parser=str,
                           get_cmd='OUTP:STAT?',
                           set_cmd='OUTP:STAT {}',
                           val_mapping=create_on_off_val_mapping(on_val='1',
                                                                 off_val='0'))
 
        self.add_parameter(name='power',
                           label='Power',
                           get_parser=float,
                           get_cmd='SOUR:POW?',
                           set_cmd='SOUR:POW {}',
                           unit='dBm',
                           vals=Numbers(min_value=-50,
                                        max_value=10))
 
        self.add_parameter(name='ifbw',
                           label='IF Bandwidth',
                           get_parser=float,
                           get_cmd='SENS1:BWID?',
                           set_cmd='SENS1:BWID {}',
                           unit='Hz',
                           vals=Enum(*np.append(np.kron([1, 1.5, 2, 3, 5, 7],
                                                       10 ** np.arange(5)),
                                               np.kron([1, 1.5, 2, 3], 10 ** 5)
                                               )))
 
        self.add_parameter('averaging',
                           label='Averages Status',
                           get_cmd='SENS1:AVER:STAT?',
                           set_cmd='SENS1:AVER:STAT {}',
                           val_mapping=create_on_off_val_mapping(on_val='1',
                                                                 off_val='0'))
 
        self.add_parameter('averages_trigger_enabled',
                           label='Trigger average status',
                           get_cmd='TRIG:SEQ:AVER?',
                           set_cmd='TRIG:SEQ:AVER {}',
                           val_mapping=create_on_off_val_mapping(on_val='1',
                                                                 off_val='0'))
 
        self.add_parameter('avg_num',
                           label='Averages',
                           get_cmd='SENS1:AVER:COUN?',
                           set_cmd='SENS1:AVER:COUN {}',
                           get_parser=int,
                           set_parser=int,
                           unit='',
                           vals=Numbers(min_value=1, max_value=999))
 
        self.add_parameter('electrical_delay',
                           label='Electrical delay',
                           get_cmd='CALC1:CORR:EDEL:TIME?',
                           set_cmd='CALC1:CORR:EDEL:TIME {}',
                           get_parser=float,
                           set_parser=float,
                           unit='s',
                           vals=Numbers(-10, 10))
 
        self.add_parameter('electrical_distance',
                           label='Electrical distance',
                           get_cmd='CALC1:CORR:EDEL:DIST?',
                           set_cmd='CALC1:CORR:EDEL:DIST {}',
                           get_parser=float,
                           set_parser=float,
                           unit='m',
                           vals=Numbers())
 
        self.add_parameter('clock_source',
                           label='Clock source',
                           get_cmd='SENSe1:ROSCillator:SOURce?',
                           set_cmd='SENSe1:ROSCillator:SOURce {}',
                           get_parser=str,
                           set_parser=str,
                           vals = Enum('int', 'Int', 'INT',
                                       'internal', 'Internal', 'INTERNAL',
                                       'ext', 'Ext', 'EXT',
                                       'external', 'External', 'EXTERNAL'))
 
        self.add_parameter(name='fstart',
                           label='Start Frequency',
                           get_parser=float,
                           get_cmd='SENS1:FREQ:STAR?',
                           set_cmd=self._set_start,
                           unit='Hz',
                           vals=Numbers(min_value=300e3,
                                        max_value=18e9-1))
 
        self.add_parameter(name='fstop',
                           label='Stop Frequency',
                           get_parser=float,
                           get_cmd='SENS1:FREQ:STOP?',
                           set_cmd=self._set_stop,
                           unit='Hz',
                           vals=Numbers(min_value=300e3+1,
                                        max_value=18e9))
 
        self.add_parameter(name='fcenter',
                           label='Center Frequency',
                           get_parser=float,
                           get_cmd='SENS1:FREQ:CENT?',
                           set_cmd=self._set_center,
                           unit='Hz',
                           vals=Numbers(min_value=100e3+1,
                                        max_value=18e9-1))
 
        self.add_parameter(name='fspan',
                           label='Frequency Span',
                           get_parser=float,
                           get_cmd='SENS1:FREQ:SPAN?',
                           set_cmd=self._set_span,
                           unit='Hz',
                           vals=Numbers(min_value=0,
                                        max_value=18e9-1))
 
        self.add_parameter('num_points',
                           label='Number of points',
                           get_parser=int,
                           set_parser=int,
                           get_cmd='SENS1:SWE:POIN?',
                           set_cmd=self._set_npts,
                           unit='',
                           vals=Ints(min_value=2,
                                        max_value=200001))
 
        self.add_parameter('nb_traces',
                           label='Number of traces',
                           get_parser=int,
                           set_parser=int,
                           get_cmd='CALC1:PAR:COUN?',
                           set_cmd='CALC1:PAR:COUN {}',
                           unit='',
                           vals=Ints(min_value=1,
                                     max_value=16))
 
        self.add_parameter(name='trigger_source',
                           label='Trigger source',
                           get_parser=str,
                           get_cmd=self._get_trigger,
                           set_cmd=self._set_trigger,
                           vals = Enum('bus', 'external', 'internal', 'manual',
                                       'BUS', 'EXTERNAL', 'INTERNAL', 'MANUAL',
                                       'EXT', 'INT', 'MAN', # so works with the vna.py driver
                                       'ext', 'int', 'man'))
 
 
        self.add_parameter(name='data_transfer_format',
                           label='Data format during transfer',
                           get_parser=str,
                           get_cmd='FORM:DATA?',
                           set_cmd='FORM:DATA {}',
                           vals = Enum('ascii', 'real', 'real32'))
 
        self.add_parameter(name='s11',
                           start=self.fstart(),
                           stop=self.fstop(),
                           npts=self.num_points(),
                           parameter_class=FrequencySweepMagPhase)
 
        self.add_parameter(name='s12',
                           start=self.fstart(),
                           stop=self.fstop(),
                           npts=self.num_points(),
                           parameter_class=FrequencySweepMagPhase)
 
        self.add_parameter(name='s21',
                           start=self.fstart(),
                           stop=self.fstop(),
                           npts=self.num_points(),
                           parameter_class=FrequencySweepMagPhase)
 
        self.add_parameter(name='s22',
                           start=self.fstart(),
                           stop=self.fstop(),
                           npts=self.num_points(),
                           parameter_class=FrequencySweepMagPhase)
 
        self.add_parameter(name='point_s11',
                           parameter_class=PointMagPhase)
 
        self.add_parameter(name='point_s12',
                           parameter_class=PointMagPhase)
 
        self.add_parameter(name='point_s21',
                           parameter_class=PointMagPhase)
 
        self.add_parameter(name='point_s22',
                           parameter_class=PointMagPhase)
 
        self.add_parameter(name='point_s11_iq',
                           parameter_class=PointIQ)
 
        self.add_parameter(name='point_s12_iq',
                           parameter_class=PointIQ)
 
        self.add_parameter(name='point_s21_iq',
                           parameter_class=PointIQ)
 
        self.add_parameter(name='point_s22_iq',
                           parameter_class=PointIQ)
 
        self.add_parameter(name="point_check_sweep_first",
            parameter_class=ManualParameter,
            initial_value=True,
            vals=Bool(),
            docstring="Parameter that enables a few commands, which are called"
            "before each get of a point_sxx parameter checking whether the vna"
            "is setup correctly. Is recommended to be True, but can be turned"
            "off if one wants to minimize overhead.",
        )
 
        # Added parameters to work with vna functions/sweeps in vna.py
 
        self.add_parameter(name='sweep_mode',
                        get_cmd=self._get_sweep_mode,
                        set_cmd=self._set_sweep_mode,
                        vals=Enum('HOLD', 'CONT', 'SING', 'GRO'))
 
        self.add_parameter(name='trigger_mode',
                        get_cmd=self._get_trigger_mode,
                        set_cmd=self._set_trigger_mode,
                        vals=Enum('CHAN', 'SWE', 'POIN', 'TRAC'))        

        self.register_active_traces()
 
        self.connect_message()
 
    def _set_trigger_mode(self, val: str) -> None:
        """Approximates Keysight's 4-state trigger_mode using this instrument's
        boolean TRIG:POIN (point-trigger on/off). 'POIN' -> ON (each trigger
        advances one point); any other value -> OFF (each trigger runs a full
        sweep). This is an approximation -- Copper Mountain only distinguishes
        point-vs-sweep trigger response, not four separate scopes."""
        self.write('TRIG:POIN {}'.format(1 if val.upper() == 'POIN' else 0))
 
    def _get_trigger_mode(self) -> str:
        return 'POIN' if self.ask('TRIG:POIN?').strip() in ('1', 'ON') else 'SWE'
 
    def _set_sweep_mode(self, val: str) -> None:
        """Approximates Keysight's 4-state sweep_mode using this instrument's
        boolean INIT:CONT. CONT/GRO -> continuous ON; HOLD/SING -> OFF.
        This is an approximation, not a faithful port -- Copper Mountain has
        no direct equivalent of Keysight's SING/GRO states."""
        self.write('INIT1:CONT {}'.format(1 if val.upper() in ('CONT', 'GRO') else 0))
 
    def _get_sweep_mode(self) -> str:
        return 'CONT' if self.ask('INIT1:CONT?').strip() in ('1', 'ON') else 'HOLD'
 
    def register_active_traces(self) -> List[str]:
        """Register submodules for every trace currently active on the VNA.
 
        The instrument server reads every registered Trace when its GUI tab
        opens, so inactive trace numbers must not be registered.  Call this
        after increasing ``nb_traces`` on the physical VNA:
 
        >>> vna.nb_traces(4)
        >>> vna.register_active_traces()
 
        Existing submodules are retained; only newly active traces are added.
 
        Returns:
            Names of the Trace submodules registered by this call.
        """
        added: List[str] = []
        for number in range(1, self.nb_traces() + 1):
            name = f"trace_{number}"
            if name not in self.submodules:
                trace = Trace(self, number=number, name=name)
                self.add_submodule(name, trace)
                added.append(name)
        return added
 
    def _set_start(self, val: float) -> None:
        """Sets the start frequency and updates linear trace parameters.
 
        Args:
            val (float): start frequency to be set
 
        Raises:
            ValueError: If start > stop
        """
        stop = self.fstop()
        if val >= stop:
            raise ValueError("Stop frequency must be larger than start "
                             "frequency.")
        self.write("SENS1:FREQ:STAR {}".format(val))
        # we get start as the vna may not be able to set it to the
        # exact value provided.
        start = self.fstart()
        if abs(val - start) >= 1:
            log.warning(
                "Could not set start to {} setting it to "
                "{}".format(val, start)
            )
        self.update_lin_traces()
 
    def _set_stop(self, val: float) -> None:
        """Sets the start frequency and updates linear trace parameters.
 
        Args:
            val (float): start frequency to be set
 
        Raises:
            ValueError: If stop < start
        """
        start = self.fstart()
        if val <= start:
            raise ValueError("Stop frequency must be larger than start "
                             "frequency.")
        self.write("SENS1:FREQ:STOP {}".format(val))
        # We get stop as the vna may not be able to set it to the
        # exact value provided.
        stop = self.fstop()
        if abs(val - stop) >= 1:
            log.warning(
                "Could not set stop to {} setting it to "
                "{}".format(val, stop)
            )
        self.update_lin_traces()
 
    def _set_span(self, val: float) -> None:
        """Sets frequency span and updates linear trace parameters.
 
        Args:
            val (float): frequency span to be set
        """
        self.write("SENS1:FREQ:SPAN {}".format(val))
        self.update_lin_traces()
 
    def _set_center(self, val: float) -> None:
        """Sets center frequency and updates linear trace parameters.
 
        Args:
            val (float): center frequency to be set
        """
        self.write("SENS1:FREQ:CENT {}".format(val))
        self.update_lin_traces()
 
    def _set_npts(self, val: int) -> None:
        """Sets number of points and updates linear trace parameters.
 
        Args:
            val (int): number of points to be set.
        """
        self.write("SENS1:SWE:POIN {}".format(val))
        self.update_lin_traces()
 
    def _get_trigger(self) -> str:
        """Gets trigger source.
 
        Returns:
            str: Trigger source.
        """
        r = self.ask('TRIG:SOUR?')
 
        if r.lower()=='int':
            return 'internal'
        elif r.lower()=='ext':
            return 'external'
        elif r.lower()=='man':
            return 'manual'
        else:
            return 'bus'
 
    def _set_trigger(self, trigger: str) -> None:
        """Sets trigger source.
 
        Args:
            trigger (str): Trigger source
        """
        self.write('TRIG:SOUR '+trigger.upper())
 
    def get_s(self): 
    # -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    #                          np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    #                          np.ndarray]:
        """
        Return all S parameters as magnitude in dB and phase in rad.
 
        Returns:
            Tuple[np.ndarray]: frequency [GHz],
            s11 magnitude [dB], s11 phase [rad],
            s12 magnitude [dB], s12 phase [rad],
            s21 magnitude [dB], s21 phase [rad],
            s22 magnitude [dB], s22 phase [rad]
        """
 
        self.write('CALC1:PAR:COUN 4') # 4 trace
        self.write('CALC1:PAR1:DEF S11') # Choose S11 for trace 1
        self.write('CALC1:PAR2:DEF S12') # Choose S12 for trace 2
        self.write('CALC1:PAR3:DEF S21') # Choose S21 for trace 3
        self.write('CALC1:PAR4:DEF S22') # Choose S22 for trace 4
        self.write('CALC1:TRAC1:FORM POL')  # Trace format
        self.write('CALC1:TRAC2:FORM POL')  # Trace format
        self.write('CALC1:TRAC3:FORM POL')  # Trace format
        self.write('CALC1:TRAC4:FORM POL')  # Trace format
        self.write('TRIG:SEQ:SING') # Trigger a single sweep
        self.ask('*OPC?') # Wait for measurement to complete
 
        # Get data as string
        freq_raw = self.ask("SENS1:FREQ:DATA?")
        s11_raw = self.ask("CALC1:TRAC1:DATA:FDAT?")
        s12_raw = self.ask("CALC1:TRAC2:DATA:FDAT?")
        s21_raw = self.ask("CALC1:TRAC3:DATA:FDAT?")
        s22_raw = self.ask("CALC1:TRAC4:DATA:FDAT?")
 
        # Get data as numpy array
        freq = np.fromstring(freq_raw, dtype=float, sep=',')
        s11 = np.fromstring(s11_raw, dtype=float, sep=',')
        s11 = s11[0::2] + 1j*s11[1::2]
        s12 = np.fromstring(s12_raw, dtype=float, sep=',')
        s12 = s12[0::2] + 1j*s12[1::2]
        s21 = np.fromstring(s21_raw, dtype=float, sep=',')
        s21 = s21[0::2] + 1j*s21[1::2]
        s22 = np.fromstring(s22_raw, dtype=float, sep=',')
        s22 = s22[0::2] + 1j*s22[1::2]
 
        return (np.array(freq), self._db(s11), np.array(np.angle(s11)),
                                self._db(s12), np.array(np.angle(s12)),
                                self._db(s21), np.array(np.angle(s21)),
                                self._db(s22), np.array(np.angle(s22)))
 
    def update_lin_traces(self) -> None:
        """
        Updates start, stop and npts of all trace parameters so that the
        setpoints and shape are updated for the sweep.
        """
        start = self.fstart()
        stop = self.fstop()
        npts = self.num_points()
        for _, parameter in self.parameters.items():
            if isinstance(parameter, (FrequencySweepMagPhase)):
                try:
                    parameter.set_sweep(start, stop, npts)
                except AttributeError:
                    pass
 
    def clear_averages(self) -> None:
        """
        Resets average count to 0
        """
        self.write("SENS1:AVER:CLE")
 
    @staticmethod
    def _db(data: np.ndarray) -> np.ndarray:
        """
        Return dB from magnitude
 
        Args:
            data (np.ndarray): data to be transformed into dB.
 
        Returns:
            data (np.ndarray): data transformed in dB.
        """
 
        return 20.*np.log10(np.abs(data))
