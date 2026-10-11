"""Saturation spectroscopy repeated across flux, around zero or half flux.

One instance measures one window.  Both tones move with the bias current: the
pump window is centred on the scqubits prediction for f01 at that flux, and the
readout tone follows the resonator notch that ResonatorSpectroscopyVsFlux
already fitted at that current.

Success is a coverage fraction rather than a fit at every point: the prediction
is only as good as EJ/EC/EL, so the edges of a window are expected to fail, and
a per-point check would abort on the first of them.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import json
import xarray as xr
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis
from labcore.data.datadict_storage import datadict_from_hdf5
from labcore.measurement.record import record_as, independent, dependent
from labcore.measurement.storage import run_and_save_sweep
from labcore.measurement.sweep import Sweep, pointer, sweep_parameter
from labcore.protocols.base import (CheckResult, Correction, CorrectionParameter,
                                    ProtocolOperation)

from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import PulseProbeSpectroscopy
from flux_offset.symmetry import extract_resonator_signal

from operations.fluxonium.fluxonium_spectrum import (
    dispersive_shift, fluxonium_f01,
)
from cqedtoolbox.protocols.operations.single_qubit.sat_spec import (
    AveragingIncreaseFactor, IncreaseAveragingCorrection, IncreasePowerCorrection,
    MaxAveragingIncreases, MaxPowerIncreases, PowerIncreaseFactor, SNRThreshold,
    SyntheticSatSpecData,
)
from parameters import (
    CouplingG, ECParam, EJParam, ELParam, EndSaturationSpecFrequency,
    HalfFluxCurrent, ReadoutFrequency, Repetition, ResonatorFr,
    SatSpecFluxSteps, SatSpecHalfFluxRange, SatSpecHalfFreqSpan,
    SatSpecZeroFluxRange, SatSpecZeroFreqSpan,
    SaturationSpecDriveGain, SaturationSpecSteps, StartSaturationSpecFrequency,
    ZeroFluxCurrent,
)

logger = logging.getLogger(__name__)

DMT_RESOLUTION = 0.125 #: Step size of the DMT current source, in uA

#: How far from the known readout frequency a notch may sit and still be taken
#: for the resonator, MHz.  Measured on 2026-10-10 the resonator moves only
#: 23 kHz across the zero-flux window and 129 kHz across the half-flux one,
#: while the spurious satellite sits ~2.7 MHz above it -- so this is roughly
#: 15x the real motion and still comfortably excludes the satellite.
PROBE_SEARCH_BAND = 2.0


# ---------------------------------------------------------------------------
# Replaying saved sweeps, so an operation can run without repeating the
# measurement that normally feeds it
# ---------------------------------------------------------------------------

def _analysis_dir(path, analysis_name: str) -> Path:
    """The analysis folder, given it, the measurement folder, or its ddh5."""
    path = Path(path)
    if path.name == "data.ddh5":
        return path.parent / analysis_name
    if path.name == analysis_name:
        return path
    return path / analysis_name


def _saved_loader(analysis_dir: Path):
    """Read one named array out of a DatasetAnalysis folder's JSON files."""
    def load(name: str) -> np.ndarray:
        files = sorted(analysis_dir.glob(f"*_{name}.json"))
        if not files:
            raise FileNotFoundError(f"No saved {name!r} analysis in {analysis_dir}")
        with files[-1].open() as file:
            return np.asarray(json.load(file)[name], float)
    return load


@dataclass
class SavedResonatorCurveSource:
    """Stands in for ``ResonatorSpectroscopyVsFlux``.

    The attribute names deliberately match that operation's, so
    `_resonator_probe` reads a live sweep and a replayed one through the same
    code without an adapter.
    """

    independents: dict          # {"flux": uA, "frequencies": MHz}
    dependents: dict            # {"signal": |S| map, (n_flux, n_freq)}
    notch_centres: np.ndarray   # (n_flux, 2), NaN-padded


def load_saved_resonator_curve(
    path: str | Path,
    *,
    analysis_name: str = "ResonatorSpectroscopyVsFlux",
) -> SavedResonatorCurveSource:
    """Replay a previous resonator-vs-flux sweep as a probe-frequency source.

    ``path`` may be the measurement folder, its ``data.ddh5``, or its analysis
    folder.  Nothing is re-fitted: `ResonatorSpectroscopyVsFlux.analyze()`
    already ran `fit_lorentzian_notches` over every trace and saved the result,
    so this only reads it back.
    """
    load = _saved_loader(_analysis_dir(path, analysis_name))
    return SavedResonatorCurveSource(
        independents={"flux": load("flux"), "frequencies": load("frequencies")},
        dependents={"signal": load("signal_magnitude")},
        notch_centres=load("notch_centres"),
    )


@dataclass
class SavedQubitCurveSource:
    """Minimal ``SaturationSpectroscopyVsFlux`` interface the qubit fit needs.

    `FluxoniumQubitTheoryFit` only reads `currents`, `peak_freq`, `found` and
    `data_loc` off its sources, so a saved sweep can stand in for a live one.
    """

    currents: np.ndarray
    peak_freq: np.ndarray
    found: np.ndarray
    data_loc: Path
    window: str


def load_saved_qubit_curve(
    path: str | Path,
    zero_current: float,
    half_current: float,
    *,
    window: str = "half",
    analysis_name: str = "SaturationSpectroscopyVsFlux",
) -> SavedQubitCurveSource:
    """Rebuild a qubit-fit source from a saved saturation-spec-vs-flux run.

    `analyze()` already stored the extracted line as `flux`, `peak_frequency`
    and `found`, so nothing is re-fitted here -- this only reads them back and
    keeps the points belonging to one flux window.

    Each point is assigned to whichever of `zero_current` / `half_current` it
    sits nearer.  That works for the older two-window datasets, where both
    windows share one file, and is a no-op for single-window ones; it assumes
    nothing about ordering, spacing, or the two halves being equal in length.

    ``path`` may be the measurement folder, its ``data.ddh5``, or its
    ``SaturationSpectroscopyVsFlux`` analysis folder -- the last of which is
    also named ``..._zero`` or ``..._half`` for sweeps taken after the split,
    so pass `analysis_name` to match.
    """
    if window not in ("zero", "half"):
        raise ValueError(f"window must be 'zero' or 'half', not {window!r}")

    analysis_dir = _analysis_dir(path, analysis_name)
    load = _saved_loader(analysis_dir)
    currents, peak_freq, found = load("flux"), load("peak_frequency"), load("found")
    if not (currents.shape == peak_freq.shape == found.shape):
        raise ValueError(
            f"saved flux{currents.shape}, peak_frequency{peak_freq.shape} and "
            f"found{found.shape} disagree in length"
        )

    centre = {"zero": zero_current, "half": half_current}[window]
    other = {"zero": half_current, "half": zero_current}[window]
    keep = np.abs(currents - centre) <= np.abs(currents - other)
    if not keep.any():
        raise ValueError(
            f"no saved point is nearer {centre} uA than {other} uA; the sweep "
            f"covers [{currents.min():.3g}, {currents.max():.3g}] uA"
        )

    return SavedQubitCurveSource(
        currents=currents[keep],
        peak_freq=peak_freq[keep],
        found=found[keep].astype(bool),
        data_loc=analysis_dir.parent,
        window=window,
    )

# ---------------------------------------------------------------------------
# Correction parameters
# ---------------------------------------------------------------------------

@dataclass
class SatSpecVsFluxMinGoodFraction(CorrectionParameter):
    name: str = field(default="sat_spec_vs_flux_min_good_fraction", init=False)
    description: str = field(default="Minimum fraction of flux points in each window that must show a qubit peak", init=False)

    def _qick_getter(self):
        return self.params.corrections.sat_spec_vs_flux.min_good_fraction()

    def _qick_setter(self, value):
        self.params.corrections.sat_spec_vs_flux.min_good_fraction(value)


@dataclass
class SatSpecVsFluxSpanFactor(CorrectionParameter):
    name: str = field(default="sat_spec_vs_flux_span_factor", init=False)
    description: str = field(default="Factor by which to widen the pump sweep span", init=False)

    def _qick_getter(self):
        return self.params.corrections.sat_spec_vs_flux.span_factor()

    def _qick_setter(self, value):
        self.params.corrections.sat_spec_vs_flux.span_factor(value)


@dataclass
class SatSpecVsFluxMaxSpanIncreases(CorrectionParameter):
    name: str = field(default="sat_spec_vs_flux_max_span_increases", init=False)
    description: str = field(default="Maximum number of span increases to try", init=False)

    def _qick_getter(self):
        return int(self.params.corrections.sat_spec_vs_flux.max_span_increases())

    def _qick_setter(self, value):
        self.params.corrections.sat_spec_vs_flux.max_span_increases(value)


class WidenSpanCorrection(Correction):
    name = "widen_span"
    description = "Widen the pump sweep around each predicted qubit frequency"
    triggered_by = "flux_coverage"

    def __init__(self, span_param, factor_param, max_increases_param):
        self.span_param = span_param
        self.factor_param = factor_param
        self.max_increases_param = max_increases_param
        self._original: float | None = None
        self._count = 0
        self._last_change = ""

    def can_apply(self) -> bool:
        return self._count < int(self.max_increases_param())

    def apply(self) -> None:
        if self._original is None:
            self._original = self.span_param()
        old = self.span_param()
        new = self._original * (self.factor_param() ** (self._count + 1))
        self.span_param(new)
        self._count += 1
        self._last_change = f"span: {old:.1f} -> {new:.1f} MHz"

    def report_output(self) -> str:
        return self._last_change


# ---------------------------------------------------------------------------
# Operation
# ---------------------------------------------------------------------------

class SaturationSpectroscopyVsFlux(ProtocolOperation):

    # In MHz, the unit the sweep works in -- unlike SaturationSpectroscopy's
    # dummy, which builds its synthetic trace in Hz.  These give a peak of
    # ~8 MHz half width, comfortably resolved inside the default 100 MHz span.
    _DUMMY_GAMMA_1 = 2.0
    _DUMMY_GAMMA_2 = 2.0
    _DUMMY_F_RABI = 8.0
    _DUMMY_NOISE_AMP = 0.01
    _DUMMY_ANGLE = np.pi / 4

    #: (current range, pump span, window centre) parameters per window.  The
    #: ranges differ because fitting EJ/EC/EL needs a wide half-flux sweep,
    #: while the spans stay independent so the zero-flux one can be narrowed
    #: against the generator's mirror image without shrinking the other.
    _WINDOW_PARAMS = {
        "zero": (SatSpecZeroFluxRange, SatSpecZeroFreqSpan, ZeroFluxCurrent),
        "half": (SatSpecHalfFluxRange, SatSpecHalfFreqSpan, HalfFluxCurrent),
    }

    def __init__(self, params, window, set_flux_current=None, source=None):
        super().__init__()
        if window not in self._WINDOW_PARAMS:
            raise ValueError(f"window must be 'zero' or 'half', not {window!r}")
        self.window = window
        self.name = f"{self.name}_{window}"      # distinct dataset per window
        self.set_flux_current = set_flux_current
        # Where the readout tone is placed at each flux point: a
        # ResonatorSpectroscopyVsFlux operation, or load_saved_resonator_curve
        # replaying one.  Not FluxOffsetInference -- that only adds the CNN
        # pipeline's own branch pick on top of the same notch fits, and the
        # zero/half currents it produces are read from the parameter manager.
        self.source = source

        range_param, span_param, centre_param = self._WINDOW_PARAMS[window]
        self._register_inputs(
            repetitions=Repetition(params),
            steps=SaturationSpecSteps(params),
            start_freq=StartSaturationSpecFrequency(params),
            end_freq=EndSaturationSpecFrequency(params),
            drive_gain=SaturationSpecDriveGain(params),
            readout_freq=ReadoutFrequency(params),
            # Both calibration points, because the flux fraction needs the
            # pair; `window_centre` aliases whichever one this window sweeps.
            zero_current=ZeroFluxCurrent(params),
            half_current=HalfFluxCurrent(params),
            window_centre=centre_param(params),
            flux_range=range_param(params),
            flux_steps=SatSpecFluxSteps(params),
            freq_span=span_param(params),
            EC=ECParam(params), EL=ELParam(params), EJ=EJParam(params),
            coupling_g=CouplingG(params), bare_fr=ResonatorFr(params),
        )
        self._register_correction_params(
            min_good_fraction=SatSpecVsFluxMinGoodFraction(params),
            span_factor=SatSpecVsFluxSpanFactor(params),
            max_span_increases=SatSpecVsFluxMaxSpanIncreases(params),
            snr_threshold=SNRThreshold(params),
            averaging_increase_factor=AveragingIncreaseFactor(params),
            max_averaging_increases=MaxAveragingIncreases(params),
            power_increase_factor=PowerIncreaseFactor(params),
            max_power_increases=MaxPowerIncreases(params),
        )

        # The two imported corrections take a window-shift strategy only to
        # call .reset() on it.  There is no fixed window here -- it is rebuilt
        # from the prediction at every flux point -- so a no-op stands in.
        no_window = SimpleNamespace(reset=lambda: None)
        self._register_check("flux_coverage", self._check_coverage, [
            WidenSpanCorrection(self.freq_span, self.span_factor,
                                self.max_span_increases),
            IncreasePowerCorrection(self.drive_gain, no_window,
                                    self.power_increase_factor,
                                    self.max_power_increases),
            IncreaseAveragingCorrection(self.repetitions, no_window,
                                        self.averaging_increase_factor,
                                        self.max_averaging_increases),
        ])

        self.condition = ("Success if a qubit peak is found in at least the "
                          "required fraction of flux points")
        # Every attempt re-measures the whole window, so the default of 100 is
        # meaningless here.
        self.max_attempts = 4

        self.independents = {"flux": [], "frequencies": []}
        self.dependents = {"signal": []}
        self.currents = None
        self.centers = None
        self.probes = None
        self.peak_freq = None
        self.snr = None
        self.found = None

    # --- planning -------------------------------------------------------------

    def _flux_fraction(self, currents):
        """Bias current -> external flux in units of the flux quantum."""
        zero, half = float(self.zero_current()), float(self.half_current())
        if zero == half:
            raise RuntimeError(
                "zero and half flux currents are identical "
                f"({zero} uA); run FluxOffsetInference before this operation."
            )
        return 0.5 * (currents - zero) / (half - zero)

    def _resonator_probe(self, currents):
        """Readout frequency at each flux point: the deepest notch lying close
        to the one ResonatorSpectroscopy already found.

        The band and the depth rule do separate jobs.  The band rejects a
        narrow spurious feature about 3 MHz above the resonator which, on the
        strongly curved baseline the readout line sits on, frequently fits as
        the *deeper* of the two; parking the tone there costs all of the qubit
        signal.  Depth then picks the strongest of whatever survives, so
        widening PROBE_SEARCH_BAND to admit the e0-e1 branch near half flux
        hands the choice between the two branches back to measured contrast.
        """
        source = self.source
        known_current = np.asarray(source.independents["flux"], float)
        frequencies = np.asarray(source.independents["frequencies"], float)
        magnitude = np.abs(np.asarray(source.dependents["signal"]))
        centres = np.atleast_2d(np.asarray(source.notch_centres, float))
        reference = float(self.readout_freq())

        anchor_current, anchor_frequency = [], []
        for index, current in enumerate(known_current):
            # NaN padding fails the comparison, so unfitted slots drop out here.
            candidates = centres[index][
                np.abs(centres[index] - reference) <= PROBE_SEARCH_BAND
            ]
            if candidates.size == 0:
                continue
            trace = magnitude[index]
            depth = [np.median(trace) - trace[np.argmin(np.abs(frequencies - f))]
                     for f in candidates]
            anchor_current.append(current)
            anchor_frequency.append(candidates[int(np.argmax(depth))])

        if len(anchor_current) < 2:
            raise RuntimeError(
                f"only {len(anchor_current)} of {len(known_current)} traces have a "
                f"notch within {PROBE_SEARCH_BAND} MHz of {reference:.3f} MHz; "
                f"check the readout frequency against the resonator sweep"
            )
        logger.info(
            f"probe curve: {len(anchor_current)}/{len(known_current)} anchors "
            f"within {PROBE_SEARCH_BAND} MHz of {reference:.3f}, spanning "
            f"{min(anchor_frequency):.3f}-{max(anchor_frequency):.3f} MHz"
        )
        # np.interp clamps outside the anchors, which is what we want: the
        # resonator moves well under a linewidth across either window.
        return np.interp(currents, anchor_current, anchor_frequency)

    def _plan_sweep(self):
        """Currents, pump centres and probe frequencies for the whole sweep."""
        n = int(self.flux_steps())
        centre = float(self.window_centre())
        half_width = float(self.flux_range()) / 2

        self.currents = np.linspace(centre - half_width, centre + half_width, n)
        self.currents = np.round(self.currents / DMT_RESOLUTION) * DMT_RESOLUTION
        # Snapping can put two requested points on the same step.  analyze()
        # groups the extractor's output by current, so duplicates would silently
        # merge two flux points into one; refuse the plan instead.
        if len(np.unique(self.currents)) != len(self.currents):
            raise RuntimeError(
                f"{n} points across {self.flux_range()} uA space the flux axis "
                f"more finely than the {DMT_RESOLUTION} uA source step; "
                f"reduce the step count or widen the range."
            )

        flux = self._flux_fraction(self.currents)
        EJ, EC, EL = self.EJ(), self.EC(), self.EL()
        f01 = np.array([fluxonium_f01(EJ, EC, EL, phi) for phi in flux])
        self.centers = 1e3 * f01

        # Don't technically need this part

        # g and fr diagnose rather than correct -- they are predictions too, and
        # a centre sitting near the resonator is unreliable however it was
        # computed.  See fluxonium_spectrum.fluxonium_f01.  Diagnosis-only here,
        # not everywhere: fluxonium_theory_fit fits both for real, and
        # flux_offset_inference reads bare_fr.
        g, fr = self.coupling_g(), self.bare_fr()
        shift = max(abs(dispersive_shift(f, g, fr)) for f in f01)
        logger.info(f"planned {len(self.currents)} flux points; f01 spans "
                    f"{f01.min():.3f}-{f01.max():.3f} GHz, dispersive shift up "
                    f"to {1e3 * shift:.2f} MHz")
        if (near := int(np.sum(np.abs(f01 - fr) < 10 * g))):
            logger.warning(f"{near} flux point(s) sit within 10g of the resonator "
                           f"at {fr:.3f} GHz; predicted centres are unreliable there")

        self.probes = self._resonator_probe(self.currents)

    # --- measurement ----------------------------------------------------------

    def _measure_qick(self) -> Path:
        if self.set_flux_current is None:
            raise RuntimeError(
                "No flux current setter was supplied. Construct this operation "
                "with set_flux_current=<callable> so it can move the flux bias."
            )
        self._plan_sweep()
        span = float(self.freq_span())
        currents, centers, probes = self.currents, self.centers, self.probes

        @pointer(independent("current", unit="uA"))
        def sweep_current():
            for value, centre, probe in zip(currents, centers, probes):
                logger.debug(f"flux {value} uA: pump {centre:.1f}, probe {probe:.1f} MHz")
                self.set_flux_current(float(value))
                self.readout_freq(float(probe))
                self.start_freq(float(centre) - span / 2)
                self.end_freq(float(centre) + span / 2)
                print("Probe at {:.3f} MHz, pump at {:.3f} MHz, flux {:.3f} uA".format(probe, centre, value))
                yield {"current": float(value)}

        # Restore your original values once you're done with the sweep
        restore = [(p, p()) for p in (self.readout_freq, self.start_freq, self.end_freq)]
        try:
            logger.info("Starting qick saturation spectroscopy vs flux measurement")
            loc, _ = run_and_save_sweep(Sweep(sweep_current) @ PulseProbeSpectroscopy(),
                                        "data", self.name)
            logger.info("Measurement complete")
            return loc
        finally:
            for param, value in restore:
                param(value)
            # self.set_flux_current(np.round(float(self.zero_current) / DMT_RESOLUTION) * DMT_RESOLUTION)

    def _measure_dummy(self) -> Path:
        """Synthetic peaks at the predicted centres, with the same layout."""
        self._plan_sweep()
        span, n_freq = float(self.freq_span()), int(self.steps())
        freq_grid = (self.centers[:, None]
                     + np.linspace(-span / 2, span / 2, n_freq)[None, :])
        flux_grid = np.broadcast_to(self.currents[:, None], freq_grid.shape)

        peak = lambda centre: SyntheticSatSpecData(
            fq=centre, delta_fr=0.0, f_rabi=self._DUMMY_F_RABI,
            gamma1=self._DUMMY_GAMMA_1, gamma2=self._DUMMY_GAMMA_2,
            angle=self._DUMMY_ANGLE, noise_amp=self._DUMMY_NOISE_AMP,
        )

        def signal():
            return np.stack([peak(c).generate(row)
                             for c, row in zip(self.centers, freq_grid)])

        sweep = (
            sweep_parameter("rep", range(int(self.repetitions())))
            @ record_as(lambda: flux_grid, independent("current"))
            @ record_as(lambda: freq_grid, independent("frequency"))
            @ record_as(signal, dependent("signal"))
        )
        loc, _ = run_and_save_sweep(sweep, "data", self.name)
        return loc

    # --- loading --------------------------------------------------------------

    def _load_data(self, frequency_key):
        """Flux, a per-point frequency window, and signal[flux, frequency]"""
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")

        data = datadict_from_hdf5(path)
        n_flux, n_freq = int(self.flux_steps()), int(self.steps())
        # ddh5 stores each field as one flat, row-appended column; `grid` folds
        # a column back into its (flux, frequency) block, transposing the qick
        # layout to match the dummy one.  The leading -1 absorbs repeats, hence
        # `.mean(0)` on the signal and `[0]` on the axes.
        def grid(name):
            values = np.asarray(data[name]["values"])
            if values.shape[-2:] == (n_freq, n_flux):
                return values.reshape((-1, n_freq, n_flux)).transpose(0, 2, 1)
            return values.reshape((-1, n_flux, n_freq))

        signal = grid("signal")
        # The two platforms store the flux axis differently: the QICK collector
        # yields one record per flux point, so `current` is written once per
        # point while `freq` and `signal` carry the whole inner sweep, whereas
        # the dummy path broadcasts all three over the full grid.
        current = np.asarray(data["current"]["values"])
        self.independents["flux"] = (
            grid("current")[0][:, 0] if current.size == signal.size
            else current.reshape(-1, n_flux)[0]
        )
        self.independents["frequencies"] = grid(frequency_key)[0]
        self.dependents["signal"] = signal.mean(0)

        if (self.independents["flux"].size != n_flux
                or self.independents["frequencies"].shape != (n_flux, n_freq)
                or self.dependents["signal"].shape != (n_flux, n_freq)):
            raise ValueError(
                f"Loaded data has shape {self.dependents['signal'].shape} "
                f"but expected ({n_flux}, {n_freq})"
            )

    def _load_data_qick(self):
        self._load_data("freq")

    def _load_data_dummy(self):
        self._load_data("frequency")

    # --- analysis -------------------------------------------------------------

    @staticmethod
    def _polarity(sweep):
        """Whether the qubit shows up as a dip or a peak in |S|, from the data."""
        scores = {}
        for polarity in ("dip", "peak"):
            table = extract_resonator_signal(sweep, polarity=polarity,
                                             max_candidates=2,
                                             double_bic_improvement=np.inf)
            scores[polarity] = (0.0 if table.empty else
                                float(np.median(table.amplitude / (4 * table.fit_rms))))
        best = max(scores, key=scores.get)
        logger.info(f"qubit line reads as a {best} "
                    f"(median SNR dip={scores['dip']:.2f}, peak={scores['peak']:.2f})")
        return best

    def _extract_peaks(self, detuning, magnitude, currents):
        """Per flux point: peak detuning and SNR, NaN/0 where nothing was found.

        Reuses the flux pipeline's extractor rather than fitting each trace
        here.  It does affine background plus one or two Lorentzians with BIC
        selection in a single pass and already rejects unconverged, relocated
        and background-hugging fits, so a trace it drops *is* the "no signal at
        this flux point" answer.

        The polarity is measured rather than assumed -- see `_polarity`.  An
        earlier version hard-coded "peak", reasoning that a probe sitting on
        the g-state resonance sees |S| rise when the qubit is driven; on the
        2026-10-10 data the line is unambiguously a dip (median SNR 0.94
        against 0.60), so that reasoning does not survive contact with the
        hardware.
        """
        sweep = xr.Dataset({"signal": xr.DataArray(
            magnitude, coords={"current": currents, "freq": detuning},
            dims=("current", "freq"),
        )})
        # Polarity is chosen once for the whole map rather than assumed: which
        # way |S| moves when the qubit is driven depends on where the probe
        # sits on the resonance, and on this setup it goes *down*.  Deciding
        # globally (one bit, from the median SNR) rather than per trace means
        # noise cannot pick the flattering sign for individual points.
        table = extract_resonator_signal(
            sweep, polarity=self._polarity(sweep),
            # Saturation spectroscopy has one qubit line per trace, so force the
            # single-Lorentzian model: an infinite BIC threshold stops the double
            # ever winning, and two candidates keep one rejected fit from
            # dropping the whole trace.  ~4x faster than the default 1-or-2 hunt.
            max_candidates=2, double_bic_improvement=np.inf,
        )

        # The table is tidy long-format -- one row per accepted resonance, and
        # no row at all for a current the extractor rejected -- so it has to be
        # scattered back onto the flux axis.  Whatever is never written stays
        # NaN/0, which is what "no signal at this flux point" means downstream.
        peak = np.full(len(currents), np.nan)
        snr = np.zeros(len(currents))
        index_of = {float(c): i for i, c in enumerate(currents)}
        for row in table.loc[
            # the strongest resonance per current
            table.groupby("current")["amplitude"].idxmax()
        ].itertuples():
            index = index_of[float(row.current)]          # table row -> flux index
            peak[index] = row.frequency
            snr[index] = (np.inf if row.fit_rms == 0
                          else abs(row.amplitude / (4 * row.fit_rms)))
        return peak, snr

    def analyze(self):
        flux = np.asarray(self.independents["flux"], float)
        frequencies = np.asarray(self.independents["frequencies"], float)
        magnitude = np.abs(self.dependents["signal"])
        self.figure_paths.clear()

        # A shared 1-D axis is what the extractor requires, and detuning is one
        # by construction: every row is linspace(centre +/- span/2, steps).
        detuning = frequencies - self.centers[:, None]
        if not np.allclose(detuning, detuning[0]):
            raise ValueError("flux points do not share a common detuning axis")

        # detuning[0], not detuning: the extractor needs one shared 1-D axis,
        # and the allclose above is what proves every row carries the same one.
        peak, self.snr = self._extract_peaks(detuning[0], magnitude, flux)
        self.peak_freq = peak + self.centers
        self.found = np.isfinite(peak) & (self.snr >= self.snr_threshold())

        with DatasetAnalysis(self.data_loc, self.name) as ds:
            ds.add(
                flux=flux, frequencies=frequencies, signal_magnitude=magnitude,
                predicted_center=self.centers, probe_frequency=self.probes,
                peak_frequency=self.peak_freq, snr=self.snr,
                found=self.found.astype(int),
            )
            ds.add_figure(self.name, fig=self._make_figure(flux, frequencies, magnitude))
            image_path = ds._new_file_path(ds.savefolders[1], self.name, suffix="png")
            self.figure_paths.append(image_path)

    def _make_figure(self, flux, frequencies, magnitude):
        """Absolute frequency against flux current.

        The pump window moves with flux, so each row of the mesh sits at its own
        frequencies and the map is a slanted band rather than a rectangle.
        pcolormesh takes the 2-D coordinate grids directly, so no interpolation
        onto a common axis is needed.
        """
        figure, axis = plt.subplots(figsize=(7, 5))
        currents = np.broadcast_to(flux[:, None], magnitude.shape)
        mesh = axis.pcolormesh(currents, frequencies, magnitude,
                               shading="nearest", cmap="magma", rasterized=True)
        figure.colorbar(mesh, ax=axis, label="|S| (a.u.)")
        axis.plot(flux, self.centers, color="cyan", lw=1, ls="--",
                  label="predicted f01")
        axis.scatter(flux[self.found], self.peak_freq[self.found],
                     s=12, c="lime", linewidths=0, label="fitted peak")
        axis.set(xlabel="Flux current (uA)", ylabel="Frequency (MHz)",
                 title=f"{self.window} flux")
        axis.legend(fontsize=8)
        figure.tight_layout()
        return figure

    def _check_coverage(self) -> CheckResult:
        threshold = self.min_good_fraction()
        hits, total = int(self.found.sum()), self.found.size
        return CheckResult(
            "flux_coverage", hits / total >= threshold,
            f"{hits}/{total} ({hits / total:.0%}) {self.window}-flux points showed "
            f"a qubit peak (need {threshold:.0%})",
        )
