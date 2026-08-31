"""Saturation spectroscopy repeated across flux, near zero and half flux.

Both tones move with the bias current.  The pump window is centred on the
scqubits prediction for f01 at that flux; the readout tone follows the
resonator's measured dressed frequency, which `FluxOffsetInference` already
extracted from the resonator-vs-flux sweep and left on its `FluxPoints`.

Success is a coverage fraction per window rather than a fit at every point:
the prediction is only as good as EJ/EC/EL, so the edges of a window are
expected to fail, and a per-point check would abort on the first of them.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
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
from cqedtoolbox.protocols.flux_offset.symmetry import extract_resonator_signal
from cqedtoolbox.protocols.operations.fluxonium.fluxonium_spectrum import (
    dispersive_shift, fluxonium_f01,
)
from cqedtoolbox.protocols.operations.single_qubit.sat_spec import (
    AveragingIncreaseFactor, IncreaseAveragingCorrection, IncreasePowerCorrection,
    MaxAveragingIncreases, MaxPowerIncreases, PowerIncreaseFactor, SNRThreshold,
    SyntheticSatSpecData,
)
from cqedtoolbox.protocols.parameters import (
    CouplingG, ECParam, EJParam, ELParam, EndSaturationSpecFrequency,
    HalfFluxCurrent, ReadoutFrequency, Repetition, ResonatorFr,
    SatSpecFluxFreqSpan, SatSpecFluxRange, SatSpecFluxSteps,
    SaturationSpecDriveGain, SaturationSpecSteps, StartSaturationSpecFrequency,
    ZeroFluxCurrent,
)

logger = logging.getLogger(__name__)

#: Step size of the DMT current source, in uA.  Requested currents are snapped
#: to it so the recorded axis matches what the hardware actually applied.
DMT_RESOLUTION = 0.125


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
        self._last_change = f"span: {old:.1f} → {new:.1f} MHz"

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

    def __init__(self, params, set_flux_current=None, source=None):
        super().__init__()
        self.params = params
        self.set_flux_current = set_flux_current
        # Needed for one thing only: the resonator frequency curve, a vector
        # with no home in the parameter manager.  The window centres come from
        # the parameter manager, which FluxOffsetInference has already written.
        self.source = source

        self._register_inputs(
            repetitions=Repetition(params),
            steps=SaturationSpecSteps(params),
            start_freq=StartSaturationSpecFrequency(params),
            end_freq=EndSaturationSpecFrequency(params),
            drive_gain=SaturationSpecDriveGain(params),
            readout_freq=ReadoutFrequency(params),
            zero_current=ZeroFluxCurrent(params),
            half_current=HalfFluxCurrent(params),
            flux_range=SatSpecFluxRange(params),
            flux_steps=SatSpecFluxSteps(params),
            freq_span=SatSpecFluxFreqSpan(params),
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

        # The two imported corrections reset a window-shift strategy that has no
        # counterpart here: the window is recomputed from scratch at every flux
        # point, so there is nothing to restore.
        no_window = SimpleNamespace(reset=lambda: None)
        # One check covering both windows, not one per window: correct() applies
        # a correction per *failed check*, so two checks sharing these
        # strategies would widen the span twice in a single attempt.
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
                          "required fraction of flux points in *each* window")
        # Every attempt re-runs the whole two-window sweep, so the default of
        # 100 is meaningless here.
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
        """The measured dressed resonator frequency at each current."""
        result = getattr(self.source, "result", None)
        if result is None or result.dominant_freq is None:
            raise RuntimeError(
                "No resonator frequency curve available. Construct this "
                "operation with source=<FluxOffsetInference instance> that has "
                "run far enough to extract the dominant resonance."
            )
        # dominant_freq is in the units of the Dataset FluxOffsetInference built,
        # i.e. native scaled by freq_to_ghz; undo that to get back to MHz.
        known_current = np.asarray(result.dominant_current, float)
        known_freq = np.asarray(result.dominant_freq, float) / self.source.freq_to_ghz
        lo, hi = known_current.min(), known_current.max()
        if currents.min() < lo or currents.max() > hi:
            raise RuntimeError(
                f"flux windows span [{currents.min():.3g}, {currents.max():.3g}] uA "
                f"but the resonator curve only covers [{lo:.3g}, {hi:.3g}] uA; "
                f"np.interp would silently clamp at the edges."
            )
        return np.interp(currents, known_current, known_freq)

    def _plan_sweep(self):
        """Currents, pump centres and probe frequencies for the whole sweep."""
        n = int(self.flux_steps())
        half_width = float(self.flux_range()) / 2
        # Not sorted: the first n points are the zero-flux window and the second
        # n the half-flux window, which is how the coverage check splits them.
        self.currents = np.concatenate([
            np.linspace(centre - half_width, centre + half_width, n)
            for centre in (float(self.zero_current()), float(self.half_current()))
        ])
        self.currents = np.round(self.currents / DMT_RESOLUTION) * DMT_RESOLUTION

        flux = self._flux_fraction(self.currents)
        EJ, EC, EL = self.EJ(), self.EC(), self.EL()
        f01 = np.array([fluxonium_f01(EJ, EC, EL, phi) for phi in flux])
        self.centers = 1e3 * f01

        # g and fr diagnose rather than correct -- they are predictions too, and
        # a centre sitting near the resonator is unreliable however it was
        # computed.  See fluxonium_spectrum.fluxonium_f01.
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
                # Read back by QickConfig.config_ when the inner sweep starts,
                # which QickBoardSweep.setup re-runs once per flux point.
                self.readout_freq(float(probe))
                self.start_freq(float(centre) - span / 2)
                self.end_freq(float(centre) + span / 2)
                yield {"current": float(value)}

        # Live tuning parameters owned by other operations; they must not be
        # left holding the last flux point's values.
        restore = {p: p() for p in (self.readout_freq, self.start_freq, self.end_freq)}
        try:
            logger.info("Starting qick saturation spectroscopy vs flux measurement")
            loc, _ = run_and_save_sweep(Sweep(sweep_current) @ PulseProbeSpectroscopy(),
                                        "data", self.name)
            logger.info("Measurement complete")
            return loc
        finally:
            for param, value in restore.items():
                param(value)
            self.set_flux_current(float(self.zero_current()))

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
        """Flux, a per-point frequency window, and signal[flux, frequency].

        Deliberately not ResonatorSpectroscopyVsFlux's loader: that one collapses
        the frequency axis to a single shared grid, and here every flux point has
        its own window, so the axis is genuinely 2-D and neither axis is sorted.
        """
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")

        data = datadict_from_hdf5(path)
        shape = 2 * int(self.flux_steps()), int(self.steps())
        grid = lambda name: np.asarray(data[name]["values"]).reshape((-1, *shape))[0]

        self.independents["flux"] = grid("current")[:, 0]
        self.independents["frequencies"] = grid(frequency_key)
        self.dependents["signal"] = (
            np.asarray(data["signal"]["values"]).reshape((-1, *shape)).mean(0)
        )

    def _load_data_qick(self):
        self._load_data("freq")

    def _load_data_dummy(self):
        self._load_data("frequency")

    # --- analysis -------------------------------------------------------------

    def _extract_peaks(self, detuning, magnitude, currents):
        """Per flux point: peak detuning and SNR, NaN/0 where nothing was found.

        Reuses the flux pipeline's extractor rather than fitting each trace
        here.  It does affine background plus one or two Lorentzians with BIC
        selection in a single pass and already rejects unconverged, relocated
        and background-hugging fits, so a trace it drops *is* the "no signal at
        this flux point" answer.

        `polarity="peak"` and not both polarities: the probe was placed on the
        g-state dressed resonance, so driving the qubit moves the resonance away
        from it and |S| rises.  A second "dip" pass would only give noise a
        chance to outscore the real feature and inflate the coverage fraction.
        """
        sweep = xr.Dataset({"signal": xr.DataArray(
            magnitude, coords={"current": currents, "freq": detuning},
            dims=("current", "freq"),
        )})
        table = extract_resonator_signal(sweep, polarity="peak")

        peak = np.full(len(currents), np.nan)
        snr = np.zeros(len(currents))
        index_of = {float(c): i for i, c in enumerate(currents)}
        # One row per current: the strongest resonance, as dominant_series does.
        for row in table.loc[table.groupby("current")["amplitude"].idxmax()].itertuples():
            index = index_of[float(row.current)]
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
            ds.add_figure(self.name, fig=self._make_figure(detuning[0], magnitude))
            image_path = ds._new_file_path(ds.savefolders[1], self.name, suffix="png")
            self.figure_paths.append(image_path)

    def _make_figure(self, detuning, magnitude):
        """One panel per window, on the detuning axis so the maps stay rectangular."""
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        for axis, (label, window) in zip(axes, self._windows()):
            mesh = axis.pcolormesh(self.currents[window], detuning,
                                   magnitude[window].T, shading="auto",
                                   cmap="magma", rasterized=True)
            figure.colorbar(mesh, ax=axis, label="|S| (a.u.)")
            axis.axhline(0, color="cyan", lw=1, ls="--", label="predicted f01")
            fitted = window & self.found
            axis.scatter(self.currents[fitted],
                         (self.peak_freq - self.centers)[fitted],
                         s=12, c="lime", linewidths=0, label="fitted peak")
            axis.set(xlabel="Flux current (uA)",
                     ylabel="Detuning from predicted f01 (MHz)",
                     title=f"{label} flux")
            axis.legend(fontsize=8)
        figure.tight_layout()
        return figure

    def _windows(self):
        """(label, boolean mask) for the zero- and half-flux halves of the sweep."""
        n = int(self.flux_steps())
        for label, start in (("Zero", 0), ("Half", n)):
            mask = np.zeros(2 * n, bool)
            mask[start:start + n] = True
            yield label, mask

    def _check_coverage(self) -> CheckResult:
        """Both windows must independently clear the threshold.

        One check rather than one per window so that a single correction is
        applied per attempt, but the fractions stay separate: a window that
        failed outright must not be able to hide behind a good one.
        """
        threshold = self.min_good_fraction()
        parts, passed = [], True
        for label, window in self._windows():
            hits, total = int(self.found[window].sum()), int(window.sum())
            passed &= hits / total >= threshold
            parts.append(f"{label.lower()} flux {hits}/{total} ({hits / total:.0%})")
        return CheckResult(
            "flux_coverage", passed,
            f"{', '.join(parts)} showed a qubit peak (need {threshold:.0%} in each)",
        )
