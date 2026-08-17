import logging
from pathlib import Path
from dataclasses import dataclass, field

import lmfit
import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

from labcore.analysis import DatasetAnalysis, FitResult
from labcore.measurement.storage import run_and_save_sweep
from labcore.data.datadict_storage import datadict_from_hdf5, load_as_xr
from labcore.measurement import sweep_parameter, record_as

from labcore.protocols.base import (ProtocolOperation, OperationStatus, serialize_fit_params,
                                    ParamImprovement, CorrectionParameter, CheckResult, Correction)
from cqedtoolbox.protocols.parameters import (Repetition,
                                              ResonatorSpecSteps, ReadoutGain, ReadoutLength, StartReadoutFrequency,
                                              EndReadoutFrequency, ReadoutFrequency, nestedAttributeFromString)
from cqedtoolbox.measurement_lib.opx.advanced.qubit_tuneup import measure_pulse_resonator_spec
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import FreqSweepProgram

from cqedtoolbox.fitfuncs.resonators import HangerResponseBruno


logger = logging.getLogger(__name__)


@dataclass
class UnwindAndFitRet:
    signal_unwind: ArrayLike
    magnitude: ArrayLike
    phase: ArrayLike
    fit_curve: ArrayLike
    fit_result: FitResult
    residuals: ArrayLike
    snr: float
    linewidth: float
    rejection_reason: str | None
    fig: plt.Figure
    ax: plt.Axes

@dataclass
class SyntheticHangerResonatorData:
    f0: float
    Qc: float
    Qi: float
    A: float
    phi: float
    noise_amp: float
    
    def generate(self, frequencies: ArrayLike) -> ArrayLike:
        Q_l = 1./(1./self.Qc + 1./self.Qi)
        Q_e_complex = self.Qc * np.exp(-1j * self.phi)
        response = self.A * (1 - (Q_l / Q_e_complex) / (1 + 2j * Q_l * (frequencies - self.f0) / self.f0))
        return response + self.noise_amp * (np.random.randn() + 1j * np.random.randn())


@dataclass
class SNRThreshold(CorrectionParameter):
    name: str = field(default="resonator_spec_SNR_threshold", init=False)
    description: str = field(default="SNR threshold", init=False)

    def _qick_getter(self):
        return self.params.corrections.res_spec.snr()

    def _qick_setter(self, value):
        self.params.corrections.res_spec.snr(value)

    def _opx_getter(self):
        return self.params.corrections.res_spec.snr()

    def _opx_setter(self, value):
        self.params.corrections.res_spec.snr(value)

@dataclass
class MaxWindowShifts(CorrectionParameter):
    name: str = field(default="res_spec_max_window_shifts", init=False)
    description: str = field(default="Number of ±n window shifts to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.res_spec.max_window_shifts())

    def _qick_setter(self, value):
        self.params.corrections.res_spec.max_window_shifts(value)

    def _opx_getter(self):
        return int(self.params.corrections.res_spec.max_window_shifts())

    def _opx_setter(self, value):
        self.params.corrections.res_spec.max_window_shifts(value)

@dataclass
class SamplingIncreaseFactor(CorrectionParameter):
    name: str = field(default="res_spec_sampling_increase_factor", init=False)
    description: str = field(default="Factor by which to increase frequency steps", init=False)

    def _qick_getter(self):
        return self.params.corrections.res_spec.sampling_factor()

    def _qick_setter(self, value):
        self.params.corrections.res_spec.sampling_factor(value)

    def _opx_getter(self):
        return self.params.corrections.res_spec.sampling_factor()

    def _opx_setter(self, value):
        self.params.corrections.res_spec.sampling_factor(value)

@dataclass
class MaxSamplingIncreases(CorrectionParameter):
    name: str = field(default="res_spec_max_sampling_increases", init=False)
    description: str = field(default="Maximum number of sampling rate increases to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.res_spec.max_sampling_increases())

    def _qick_setter(self, value):
        self.params.corrections.res_spec.max_sampling_increases(value)

    def _opx_getter(self):
        return int(self.params.corrections.res_spec.max_sampling_increases())

    def _opx_setter(self, value):
        self.params.corrections.res_spec.max_sampling_increases(value)

@dataclass
class AveragingIncreaseFactor(CorrectionParameter):
    name: str = field(default="res_spec_averaging_increase_factor", init=False)
    description: str = field(default="Factor by which to increase repetitions", init=False)

    def _qick_getter(self):
        return self.params.corrections.res_spec.averaging_factor()

    def _qick_setter(self, value):
        self.params.corrections.res_spec.averaging_factor(value)

    def _opx_getter(self):
        return self.params.corrections.res_spec.averaging_factor()

    def _opx_setter(self, value):
        self.params.corrections.res_spec.averaging_factor(value)

@dataclass
class MaxAveragingIncreases(CorrectionParameter):
    name: str = field(default="res_spec_max_averaging_increases", init=False)
    description: str = field(default="Maximum number of averaging increases to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.res_spec.max_averaging_increases())

    def _qick_setter(self, value):
        self.params.corrections.res_spec.max_averaging_increases(value)

    def _opx_getter(self):
        return int(self.params.corrections.res_spec.max_averaging_increases())

    def _opx_setter(self, value):
        self.params.corrections.res_spec.max_averaging_increases(value)

@dataclass
class MaxFitParamError(CorrectionParameter):
    name: str = field(default="res_spec_max_fit_param_error", init=False)
    description: str = field(default="Maximum allowed fractional fit parameter error (e.g. 1.0 = 100%)", init=False)

    def _qick_getter(self):
        return self.params.corrections.res_spec.max_fit_param_error()

    def _qick_setter(self, value):
        self.params.corrections.res_spec.max_fit_param_error(value)

    def _opx_getter(self):
        return self.params.corrections.res_spec.max_fit_param_error()

    def _opx_setter(self, value):
        self.params.corrections.res_spec.max_fit_param_error(value)

class WindowShiftCorrection(Correction):
    name = "window_shift"
    description = "Shift measurement window by multiples of the original window span"
    triggered_by = "snr_check"

    def __init__(self, start_param, end_param, max_shifts_param):
        self.start_param = start_param
        self.end_param = end_param
        self.max_shifts_param = max_shifts_param
        self._original_start: float | None = None
        self._original_end: float | None = None
        self._idx = 0
        self._last_new_start: float | None = None
        self._last_new_end: float | None = None

    @staticmethod
    def _shift_multiplier(idx: int) -> int:
        """idx 0 → +1, 1 → -1, 2 → +2, 3 → -2, ..."""
        n = idx // 2 + 1
        return n if idx % 2 == 0 else -n

    def can_apply(self) -> bool:
        return self._idx < int(self.max_shifts_param()) * 2

    def apply(self) -> None:
        if self._original_start is None:
            self._original_start = self.start_param()
            self._original_end = self.end_param()
        span = self._original_end - self._original_start
        shift = self._shift_multiplier(self._idx) * span
        self._last_new_start = self._original_start + shift
        self._last_new_end = self._original_end + shift
        self.start_param(self._last_new_start)
        self.end_param(self._last_new_end)
        self._idx += 1

    def report_output(self) -> str:
        if self._last_new_start is None:
            return ""
        return (f"[{self._last_new_start:.4f}, {self._last_new_end :.4f}] MHz"
                f" (shift={(self._last_new_start - self._original_start):+.1f} MHz)")

    def reset(self) -> None:
        """Restore original window and reset index. Called by higher-level corrections."""
        if self._original_start is not None:
            self.start_param(self._original_start)
            self.end_param(self._original_end)
        self._idx = 0


class IncreaseSamplingRateCorrection(Correction):
    name = "increase_sampling_rate"
    description = "Increase frequency step count and reset window shift"
    triggered_by = "snr_check"

    def __init__(self, steps_param, window_correction: WindowShiftCorrection,
                 factor_param, max_increases_param):
        self.steps_param = steps_param
        self.window_correction = window_correction
        self.factor_param = factor_param
        self.max_increases_param = max_increases_param
        self._original_steps: int | None = None
        self._count = 0
        self._last_change: str = ""

    def can_apply(self) -> bool:
        return self._count < int(self.max_increases_param())

    def apply(self) -> None:
        if self._original_steps is None:
            self._original_steps = int(self.steps_param())
        factor = self.factor_param()
        old = int(self.steps_param())
        new = int(self._original_steps * (factor ** (self._count + 1)))
        self.steps_param(new)
        self._count += 1
        self._last_change = f"steps: {old} → {new}"
        self.window_correction.reset()

    def report_output(self) -> str:
        return self._last_change


class IncreaseAveragingCorrection(Correction):
    name = "increase_averaging"
    description = "Increase repetitions and reset window shift"
    triggered_by = "snr_check"

    def __init__(self, reps_param, window_correction: WindowShiftCorrection,
                 factor_param, max_increases_param):
        self.reps_param = reps_param
        self.window_correction = window_correction
        self.factor_param = factor_param
        self.max_increases_param = max_increases_param
        self._original_reps: int | None = None
        self._count = 0
        self._last_change: str = ""

    def can_apply(self) -> bool:
        return self._count < int(self.max_increases_param())

    def apply(self) -> None:
        if self._original_reps is None:
            self._original_reps = int(self.reps_param())
        factor = self.factor_param()
        old = int(self.reps_param())
        new = int(self._original_reps * (factor ** (self._count + 1)))
        self.reps_param(new)
        self._count += 1
        self._last_change = f"reps: {old} → {new}"
        self.window_correction.reset()

    def report_output(self) -> str:
        return self._last_change


class ResonatorSpectroscopy(ProtocolOperation):

    _SIM_F0 = 7e9
    _SIM_QI = 20e3
    _SIM_QC = 20e3
    _SIM_A = 4.0
    _SIM_PHI = 0.0
    _SIM_NOISE_AMP = 0.05

    
    def __init__(self, params):
        super().__init__()
        self.params = params

        self._register_inputs(
            repetitions=Repetition(params),
            steps=ResonatorSpecSteps(params),
            gain=ReadoutGain(params),
            length=ReadoutLength(params),
            start_frequency=StartReadoutFrequency(params),
            end_frequency=EndReadoutFrequency(params),
        )
        self._register_outputs(
            readout_freq=ReadoutFrequency(params),
        )

        self._register_correction_params(
            snr_threshold=SNRThreshold(params),
            max_window_shifts=MaxWindowShifts(params),
            sampling_increase_factor=SamplingIncreaseFactor(params),
            max_sampling_increases=MaxSamplingIncreases(params),
            averaging_increase_factor=AveragingIncreaseFactor(params),
            max_averaging_increases=MaxAveragingIncreases(params),
            max_fit_param_error=MaxFitParamError(params),
        )

        self.params = params

        self._window_shift = WindowShiftCorrection(
            self.start_frequency,
            self.end_frequency,
            self.max_window_shifts,
        )
        self._increase_sampling = IncreaseSamplingRateCorrection(
            self.steps,
            self._window_shift,
            self.sampling_increase_factor,
            self.max_sampling_increases,
        )
        self._increase_averaging = IncreaseAveragingCorrection(
            self.repetitions,
            self._window_shift,
            self.averaging_increase_factor,
            self.max_averaging_increases,
        )

        self._register_check(
            "quality_check",
            self._check_quality,
            [self._increase_averaging, self._window_shift, self._increase_sampling],
        )

        self._register_success_update(
            self.readout_freq,
            lambda: self.fit_result.params["f_0"].value,
        )

        self._register_success_update(
            self.start_frequency,
            lambda: self.fit_result.params["f_0"].value - 5,
        )

        self._register_success_update(
            self.end_frequency,
            lambda: self.fit_result.params["f_0"].value + 5,
        )

        self.condition = f"Success if the SNR of the measurement is bigger than the current threshold of " # {self.SNR_THRESHOLD}"

        self.independents = {"frequencies": []}
        self.dependents = {"signal": []}
        self.unwind_signal = None
        self.magnitude = None
        self.phase = None
        self.snr = None
        self.fit_result = None
        self.fit_diagnostics = None
        self.improvements = None

    def _measure_qick(self) -> Path:
        logger.info("Starting qick resonator spectroscopy measurement")

        sweep = FreqSweepProgram()
        logger.debug("Sweep created, running measurement")
        loc, da = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Measurement complete")

        return loc

    def _measure_opx(self) -> Path:
        logger.info("Starting opx resonator spectroscopy measurement")
        loc = measure_pulse_resonator_spec()
        logger.info("Measurement complete")
        return loc
    
    def _measure_dummy(self):
        logger.info("Starting dummy resonator spectroscopy measurement")
        frequencies = np.linspace(self.start_frequency(), self.end_frequency(), int(self.steps()))
        generator = SyntheticHangerResonatorData(
            f0 = self._SIM_F0,
            Qi = self._SIM_QI,
            Qc = self._SIM_QC,
            A = self._SIM_A,
            phi = self._SIM_PHI,
            noise_amp = self._SIM_NOISE_AMP
        )

        sweep = sweep_parameter("frequencies", frequencies + self.readout_lo(), record_as(generator.generate, "signal"))
        loc, _ = run_and_save_sweep(sweep, "data", self.name)

        logger.info("Dummy measurement complete")
        return loc

    @staticmethod
    def add_mag_and_unwind_and_fit(frequencies, signal_raw, fig_title="") -> UnwindAndFitRet:
        frequencies = np.asarray(frequencies, dtype=float)
        signal_raw = np.asarray(signal_raw, dtype=complex)
        order = np.argsort(frequencies)
        frequencies = frequencies[order]
        signal_raw = signal_raw[order]
        if frequencies.size < 25:
            raise ValueError("Resonator spectroscopy needs at least 25 frequency points")

        phase_unwrap = np.unwrap(np.angle(signal_raw))
        phase_slope = np.polyfit(frequencies, phase_unwrap, 1)[0]

        signal_unwind = signal_raw * np.exp(-1j * frequencies * phase_slope)
        magnitude = np.abs(signal_raw)
        phase = np.arctan2(signal_unwind.imag, signal_unwind.real)

        span = float(frequencies[-1] - frequencies[0])
        step = float(np.median(np.diff(frequencies)))

        # median filter removes noise patterns larger than the resonator feature
        baseline = median_filter(magnitude, size=int(np.clip(magnitude.size // 8, 21, 201)) | 1, mode="nearest")
        deviation = baseline - magnitude # positive for a dip now
        peak = int(np.argmax(deviation))
        depth = float(deviation[peak])
        half = np.flatnonzero(deviation >= depth / 2) # pts outside of the feature
        kappa = max(float(frequencies[half.max()] - frequencies[half.min()]), 2 * step)

        f0_guess = float(frequencies[peak])
        base_level = max(float(baseline[peak]), np.finfo(float).eps) # level the trace would have without the resonator (A)
        relative_depth = float(np.clip(depth / base_level, 0.02, 0.95)) # = Q_l / Q_c
        q_l = f0_guess / kappa
        q_min = f0_guess / span # a linewidth wider than the scan is not a resonance

        fit_params = {
            "f_0": lmfit.Parameter("f_0", value=f0_guess, min=frequencies[0], max=frequencies[-1]),
            "A": lmfit.Parameter("A", value=base_level),
            "Q_i": lmfit.Parameter("Q_i", value=float(np.clip(q_l / (1 - relative_depth), q_min, 1e8)),
                                   min=q_min, max=1e8),
            "Q_e_mag": lmfit.Parameter("Q_e_mag", value=float(np.clip(q_l / relative_depth, q_min, 1e8)), 
                                       min=q_min, max=1e8),
        }

        fit = HangerResponseBruno(frequencies, signal_unwind)
        fit_result = fit.run(params=fit_params)
        fit_curve = fit_result.eval()
        residuals = signal_unwind - fit_curve

        params = fit_result.params
        q_i = params["Q_i"].value
        q_e = params["Q_e_mag"].value
        cos_theta = float(np.cos(params["theta"].value))
        if q_i > 0 and q_e > 0 and cos_theta > 0:
            q_c = q_e / cos_theta
            q_loaded = 1.0 / (1.0 / q_i + 1.0 / q_c)
            linewidth = float(params["f_0"].value / q_loaded)
        else:
            linewidth = float("nan")

        # ensure that the predicted/fitted resonator and linewidth are actually in the scan
        f_0 = float(params["f_0"].value)
        reasons = []
        if not frequencies[0] + step < f_0 < frequencies[-1] - step:
            reasons.append("f_0 at sweep edge")
        if not np.isfinite(linewidth) or not 2 * step <= linewidth <= 0.25 * span:
            reasons.append("linewidth unresolved or wider than a quarter of the span")
        rejection_reason = "; ".join(reasons) if reasons else None

        # noise from point to point scatter. Median keeps resonator feature out of estiamte
        noise = 1.4826 * float(np.median(np.abs(np.diff(magnitude)))) / np.sqrt(2)
        snr = depth / noise if noise > 0 else (float("inf") if depth > 0 else 0.0)

        fig, ax = plt.subplots()
        ax.set_title(fig_title)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Magnitude Signal (A.U)")
        ax.plot(frequencies, magnitude, label="Data")
        ax.plot(frequencies, np.abs(fit_curve), label="Fit")
        ax.legend()

        ret = UnwindAndFitRet(
            signal_unwind=signal_unwind,
            magnitude=magnitude,
            phase=phase,
            fit_curve=fit_curve,
            fit_result=fit_result,
            residuals=residuals,
            snr=snr,
            linewidth=linewidth,
            rejection_reason=rejection_reason,
            fig=fig,
            ax=ax,
        )

        return ret

    def _load_data_qick(self):
        path = self.data_loc/"data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)

        self.independents["frequencies"] = data["freq"]["values"]
        self.dependents["signal"] = data["signal"]["values"]

    def _load_data_opx(self):
        data = load_as_xr(self.data_loc).mean("repetition")
        q = nestedAttributeFromString(self.params, "active.qubit")()
        lo = nestedAttributeFromString(self.params, f"{q}.readout.LO")()
        self.independents["frequencies"] = data["ssb_frequency"].values + lo
        self.dependents["signal"] = data["signal_Re"].values + 1j * data["signal_Im"].values

    def _load_data_dummy(self):
        path = self.data_loc/"data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)

        self.independents["frequencies"] = data["frequencies"]["values"]
        self.dependents["signal"] = data["signal"]["values"]

    def analyze(self):
        with DatasetAnalysis(self.data_loc, self.name) as ds:
            ret = self.add_mag_and_unwind_and_fit(self.independents["frequencies"],
                                                  self.dependents["signal"],
                                                  "Resonator Spectroscopy")

            self.magnitude = ret.magnitude
            self.phase = ret.phase
            self.snr = ret.snr
            self.fit_result = ret.fit_result
            self.fit_diagnostics = ret

            ds.add(fit_curve=ret.fit_curve,
                   fit_result=ret.fit_result,
                   params=serialize_fit_params(ret.fit_result.params),
                   snr=float(ret.snr),
                   linewidth=float(ret.linewidth),
                   rejection_reason=ret.rejection_reason or "")
            ds.add_figure(self.name, fig=ret.fig)

            image_path = ds._new_file_path(ds.savefolders[1], self.name, suffix="png")
            self.figure_paths.append(image_path)

    def _check_quality(self) -> CheckResult:
        if self.snr is None or self.fit_result is None or self.fit_diagnostics is None:
            raise RuntimeError("Fit diagnostics must be set before checking quality")

        threshold = self.snr_threshold()
        snr_passed = self.snr >= threshold

        max_error = self.max_fit_param_error()
        param = self.fit_result.params["f_0"]
        linewidth = self.fit_diagnostics.linewidth
        bad_param = None
        # f_0 is an absolute frequency, so stderr/f_0 (previous SNR calculation) is ~1e-4 even for a useless
        # fit.  The scale that decides whether we located the resonance is the
        # linewidth.
        if param.stderr is None:
            bad_param = "f_0(no stderr)"
        elif not np.isfinite(linewidth) or linewidth <= 0:
            bad_param = "invalid linewidth"
        elif abs(param.stderr / linewidth) > max_error:
            bad_param = f"f_0/linewidth({abs(param.stderr / linewidth) * 100:.0f}%)"

        fit_passed = bad_param is None and self.fit_diagnostics.rejection_reason is None
        passed = snr_passed and fit_passed

        parts = [f"feature SNR={self.snr:.3f} (threshold={threshold:.3f})",
                 f"linewidth={linewidth:.4g}"]
        if bad_param:
            parts.append(f"high-error param: {bad_param}")
        if self.fit_diagnostics.rejection_reason:
            parts.append(self.fit_diagnostics.rejection_reason)

        return CheckResult("quality_check", passed, "; ".join(parts))
