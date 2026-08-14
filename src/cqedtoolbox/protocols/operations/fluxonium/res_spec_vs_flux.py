import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from labcore.data.datadict_storage import datadict_from_hdf5
from labcore.analysis import DatasetAnalysis
from labcore.analysis.fit import Fit
from labcore.measurement.sweep import sweep_parameter, Sweep, pointer
from labcore.measurement.storage import run_and_save_sweep
from labcore.measurement.record import record_as, independent, dependent

from labcore.protocols.base import (ProtocolOperation, OperationStatus,
                                    CorrectionParameter, CheckResult, EvaluateResult)
# from cqedtoolbox.protocols.parameters import (
#     Repetition, ResonatorSpecSteps, StartReadoutFrequency, EndReadoutFrequency,
#     StartFlux, EndFlux, FluxSteps,
# )
from parameters import (
    Repetition, ResonatorSpecSteps, StartReadoutFrequency, EndReadoutFrequency,
    StartFlux, EndFlux, FluxSteps,
)
from ..single_qubit.res_spec import ResonatorSpectroscopy, UnwindAndFitRet
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import FreqSweepProgram
from cqedtoolbox.fitfuncs.resonators import moving_average

logger = logging.getLogger(__name__)


@dataclass
class ResSpecVsFluxSNRThreshold(CorrectionParameter):
    name: str = field(default="res_spec_vs_flux_snr_threshold", init=False)
    description: str = field(default="Per-flux-point SNR threshold", init=False)

    def _qick_getter(self):
        return self.params.corrections.res_spec_vs_flux.snr()

    def _qick_setter(self, v):
        self.params.corrections.res_spec_vs_flux.snr(v)

    def _opx_getter(self):
        return self.params.corrections.res_spec_vs_flux.snr()

    def _opx_setter(self, v):
        self.params.corrections.res_spec_vs_flux.snr(v)


@dataclass
class ResSpecVsFluxMinGoodFraction(CorrectionParameter):
    name: str = field(default="res_spec_vs_flux_min_good_fraction", init=False)
    description: str = field(
        default="Minimum fraction of flux points that must clear the SNR threshold",
        init=False,
    )

    def _qick_getter(self): 
        return self.params.corrections.res_spec_vs_flux.min_good_fraction()

    def _qick_setter(self, v):
        self.params.corrections.res_spec_vs_flux.min_good_fraction(v)

    def _opx_getter(self):
        return self.params.corrections.res_spec_vs_flux.min_good_fraction()

    def _opx_setter(self, v):
        self.params.corrections.res_spec_vs_flux.min_good_fraction(v)


class DoubleHangerResponseBruno(Fit):
    """For single Hanger fit of each dip, we use model from: https://arxiv.org/abs/1502.04082 - pages 6/7.
    S_g/e = A (1 + alpha * (x - f_0)/f_0) (1 - Q_l/|Q_e| exp(i \theta) / (1 + 2i Q_l (x-f_0)/f_0)) exp(i(\phi_v f_0+ phi_0))
    We assume all parameters (except fr_g and fr_e) are same for both dips since fr_g and fr_e are sent from same cable and resonator.
    And S_tot = p_g * S + (1 - p_g) * S_e """

    @staticmethod
    def model(coordinates: np.ndarray, A: float, fr_g: float, fr_e: float, p_g: float, Q_i: float, Q_e_mag: float, theta: float, phase_offset: float,
              phase_slope: float, transmission_slope: float):
        
        x = coordinates
        amp_correction_g = A * (1 + transmission_slope * (x - fr_g)/fr_g)
        amp_correction_e = A * (1 + transmission_slope * (x - fr_e)/fr_e)
        phase_correction = np.exp(1j*(phase_slope * x + phase_offset))

        if Q_e_mag == 0:
            Q_e_mag = 1e-12
        Q_e_complex = Q_e_mag * np.exp(-1j*theta)
        Q_c = 1./((1./Q_e_complex).real)
        Q_l = 1./(1./Q_c + 1./Q_i)
        response_g = 1 - Q_l / np.abs(Q_e_mag) * np.exp(1j * theta) / (1. + 2j*Q_l*(x-fr_g)/fr_g)
        response_e = 1 - Q_l / np.abs(Q_e_mag) * np.exp(1j * theta) / (1. + 2j*Q_l*(x-fr_e)/fr_e)
        
        return (p_g*response_g * amp_correction_g + (1-p_g) * response_e * amp_correction_e) * phase_correction

    @staticmethod
    def guess(coordinates, data) -> Dict[str, Any]:
        """Heuristic initial guess for the double-hanger Bruno model.
        This is Bruno's single-notch guess, extended to find a second dip and p_g.
        NOTE: to avoid errors, ensure that input arrays have length >~ 250"""

        amp = np.abs(np.concatenate((data[:data.size // 10], data[-data.size // 10:]))).mean()
        data_smooth = moving_average(data)
        depth = amp - np.abs(data)
        loc = []
        margin = max(5, len(data) // 150)
        for i in range(margin, len(data) - margin):
            window = depth[i - margin : i + margin + 1]
            if depth[i] == window.max():  # in order to find the second dip we need to find two biggest local max and avoid noise on sholder of largest dip
                loc.append(i)
        loc = np.array(loc)
        vals = depth[loc]
        order = np.argsort(vals)
        dip_g = loc[order[0]]
        dip_e = loc[order[1]]
        guess_fr_g = coordinates[dip_g]
        guess_fr_e = coordinates[dip_e]
        [guess_transmission_slope, _] = np.polyfit(coordinates[:data.size // 10], np.abs(data[:data.size // 10]), 1)
        amp_correction_g = amp * (1+guess_transmission_slope*(coordinates-guess_fr_g)/guess_fr_g)

        depth_g = amp - np.abs(data_smooth[dip_g])
        depth_e = amp - np.abs(data_smooth[dip_e])
        guess_p_g = depth_g / (depth_g + depth_e)
        width_g = np.argmin(np.abs(amp - np.abs(data_smooth) - depth_g / 2))
        kappa = 2 * np.abs(coordinates[dip_g] - coordinates[width_g])
        guess_Q_l = guess_fr_g / kappa

        [slope, _] = np.polyfit(coordinates[:data.size // 10], np.angle(data_smooth[:data.size // 10], deg=False), 1)
        phase_offset = np.angle(data_smooth[0], deg=False) - slope * (coordinates[0]-0)
        phase_correction = np.exp(1j*(slope * coordinates + phase_offset))
        correction_g = amp_correction_g * phase_correction

        guess_theta = 0.5
        guess_Q_e_mag = np.abs(-guess_Q_l * np.exp(1j*guess_theta) / (data_smooth[dip_g]/correction_g[dip_g]-1))
        eps = 1e-6
        if not np.isfinite(guess_Q_e_mag) or guess_Q_e_mag <= 0:
            guess_Q_e_mag = 5e4  # some reasonable default

        denom = np.real(1 / (guess_Q_e_mag * np.exp(-1j * guess_theta)))
        if not np.isfinite(denom) or abs(denom) < eps:
            denom = np.sign(denom) * eps
        guess_Q_c = 1.0 / denom

        denom2 = 1.0 / guess_Q_l - 1.0 / guess_Q_c
        if not np.isfinite(denom2) or abs(denom2) < eps:
            denom2 = np.sign(denom2) * eps
        guess_Q_i = 1.0 / denom2
        if not np.isfinite(guess_Q_i) or guess_Q_i <= 0:
            guess_Q_i = 1e6
        if guess_p_g < 0.5:
            guess_p_g = 1 - guess_p_g
            a = guess_fr_g
            b = guess_fr_e
            guess_fr_e = a
            guess_fr_g = b

        return dict(
            A = amp,
            fr_g = guess_fr_g,
            fr_e = guess_fr_e,
            p_g = guess_p_g,
            Q_i = guess_Q_i,
            Q_e_mag = guess_Q_e_mag,
            theta = guess_theta,
            phase_offset = phase_offset,
            phase_slope = slope,
            transmission_slope = guess_transmission_slope,
        )


def _running_median(x, window=21):
    """Local median of x over a centred window, ignoring NaNs."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    half = window // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        segment = x[lo:hi]
        out[i] = np.nanmedian(segment) if np.any(np.isfinite(segment)) else np.nan
    return out


def _robust_filter_fitted_curve(flux_axis, fr_fit, snr_list, freq_axis,
                                snr_threshold=2.0, median_window=21,
                                abs_dev_threshold=None, mad_multiplier=6.0):
    """Keep the fitted points that are trustworthy.

    Three cuts: finite and above the SNR threshold, inside the swept frequency
    range (plus a 5% margin), and within a MAD-scaled distance of a local median
    of the curve.  The last one removes single-point fit failures that survive
    the SNR cut but jump far off the smooth f_r(flux) trend.
    """
    flux_axis = np.asarray(flux_axis, dtype=float)
    fr_fit = np.asarray(fr_fit, dtype=float)
    snr_list = np.asarray(snr_list, dtype=float)
    freq_axis = np.asarray(freq_axis, dtype=float)

    basic_mask = (
        np.isfinite(flux_axis)
        & np.isfinite(fr_fit)
        & np.isfinite(snr_list)
        & (snr_list >= snr_threshold)
    )
    fmin, fmax = float(np.min(freq_axis)), float(np.max(freq_axis))
    margin = 0.05 * (fmax - fmin)
    range_mask = (fr_fit >= fmin - margin) & (fr_fit <= fmax + margin)
    mask1 = basic_mask & range_mask

    fr_tmp = fr_fit.copy()
    fr_tmp[~mask1] = np.nan
    local_med = _running_median(fr_tmp, window=median_window)
    dev = np.abs(fr_fit - local_med)

    mad = np.nanmedian(np.abs(fr_tmp - local_med))
    if not np.isfinite(mad) or mad < 1e-12:
        mad = 1e-12
    if abs_dev_threshold is None:
        abs_dev_threshold = mad_multiplier * mad

    final_mask = mask1 & (dev <= abs_dev_threshold)
    logger.info(
        f"Filter summary: total={len(fr_fit)}, pass basic={int(np.sum(mask1))}, "
        f"pass local-median={int(np.sum(final_mask))}, MAD={mad:.6g}, "
        f"abs_dev_threshold={abs_dev_threshold:.6g}"
    )
    return final_mask, local_med, dev


def _infer_period_from_curve(flux_good, fr_good, n_uniform=2000):
    """Estimate one flux period from the filtered f_r(flux) curve.

    Detrends, resamples onto a uniform grid, and takes the autocorrelation peak.
    The search is restricted to lags between a quarter and 95% of the measured
    width so that small wiggles cannot masquerade as the period.
    """
    flux_good = np.asarray(flux_good, dtype=float)
    fr_good = np.asarray(fr_good, dtype=float)
    order = np.argsort(flux_good)
    x, y = flux_good[order], fr_good[order]

    x_min, x_max = float(x.min()), float(x.max())
    width = x_max - x_min
    xu = np.linspace(x_min, x_max, n_uniform)
    yu = np.interp(xu, x, y)
    yu0 = yu - np.polyval(np.polyfit(xu, yu, 1), xu)
    yu0 = yu0 - np.mean(yu0)

    n = len(yu0)
    fft = np.fft.rfft(yu0, n=2 * n)
    ac = np.fft.irfft(fft * np.conj(fft))[:n]
    ac = ac / ac[0]
    lags = np.arange(n) * (xu[1] - xu[0])

    search = (lags >= width / 4.0) & (lags <= width * 0.95)
    if not np.any(search):
        raise ValueError("No valid period range found in the measured flux span.")
    period_est = float(lags[search][int(np.argmax(ac[search]))])
    logger.info(
        f"Auto period estimate: width={width:.6g}, period={period_est:.6g}, "
        f"approx periods={width / period_est:.3f}"
    )
    return period_est, lags, ac


def _score_one_period_window(flux_good, fr_good, x0, x1, n_points=121):
    """Score a candidate one-period window; lower is better.

    Prefers windows with enough points, no large gaps, full coverage, and a
    smooth curve after interpolation.  Gaps and missing coverage are weighted
    enormously so they dominate any roughness difference.
    """
    mask = (flux_good >= x0) & (flux_good <= x1)
    fx, fy = flux_good[mask], fr_good[mask]
    if len(fx) < 30:
        return np.inf
    order = np.argsort(fx)
    fx, fy = fx[order], fy[order]

    width = x1 - x0
    if width <= 0:
        return np.inf
    gaps = np.diff(fx)
    max_gap = np.max(gaps) if len(gaps) else np.inf
    gap_score = max_gap / width
    coverage_penalty = abs(1.0 - (fx.max() - fx.min()) / width)

    x_new = np.linspace(fx.min(), fx.max(), n_points, endpoint=False)
    y_new = np.interp(x_new, fx, fy)
    roughness = np.std(np.diff(y_new, n=2))
    return roughness + 1e7 * gap_score + 1e7 * coverage_penalty


def _auto_pick_one_period_window(flux_good, fr_good, n_points=121, n_start_trials=200):
    """Infer the period, then slide a window of that width and keep the best."""
    flux_good = np.asarray(flux_good, dtype=float)
    fr_good = np.asarray(fr_good, dtype=float)
    order = np.argsort(flux_good)
    flux_good, fr_good = flux_good[order], fr_good[order]

    period_est, lags, ac = _infer_period_from_curve(flux_good, fr_good)
    data_min, data_max = float(flux_good.min()), float(flux_good.max())
    if data_max - data_min < period_est:
        raise ValueError("Data span is smaller than one inferred period.")

    starts = np.linspace(data_min, data_max - period_est, n_start_trials)
    scores = np.asarray([
        _score_one_period_window(flux_good, fr_good, x0, x0 + period_est, n_points=n_points)
        for x0 in starts
    ])
    if not np.any(np.isfinite(scores)):
        raise ValueError("No candidate one-period window had enough fitted points.")

    best = int(np.nanargmin(scores))
    x_min = float(starts[best])
    x_max = float(x_min + period_est)
    logger.info(f"Auto-selected crop: [{x_min:.6g}, {x_max:.6g}], score={scores[best]:.6g}")
    return x_min, x_max, period_est, starts, scores, lags, ac


def _resample_one_period(flux_seg, fr_seg, n_points=121):
    """Interpolate the cropped period onto the uniform grid the CNN expects."""
    flux_seg = np.asarray(flux_seg, dtype=float)
    fr_seg = np.asarray(fr_seg, dtype=float)
    order = np.argsort(flux_seg)
    flux_seg, fr_seg = flux_seg[order], fr_seg[order]

    x_new = np.linspace(flux_seg.min(), flux_seg.max(), n_points, endpoint=False)
    y_new = np.interp(x_new, flux_seg, fr_seg)
    return x_new, y_new







def add_mag_and_unwind_and_choose_fit(frequencies, signal_raw) -> UnwindAndFitRet:
    phase_unwrap = np.unwrap(np.angle(signal_raw))
    phase_slope = np.polyfit(frequencies, phase_unwrap, 1)[0]
    signal_unwind = signal_raw * np.exp(-1j * frequencies * phase_slope)
    magnitude = np.abs(signal_raw)
    phase = np.arctan2(signal_unwind.imag, signal_unwind.real)
    # DO DOUBLE HANGER FIT
    fit = DoubleHangerResponseBruno(frequencies, signal_unwind)
    fit_result = fit.run(fit)
    fit_curve = fit_result.eval()
    residuals = signal_unwind - fit_curve
    amp = fit_result.params["A"].value
    noise = np.std(residuals)
    snr = np.abs(amp / (4 * noise))

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
    reasons = []
    # if not contained:
    #     reasons.append("no contained half-height feature")
    # if not frequencies[0] + step < f_0 < frequencies[-1] - step:
    #     reasons.append("f_0 at sweep edge")
    # if not np.isfinite(linewidth) or not 2 * step <= linewidth <= 0.25 * span:
    #     reasons.append("linewidth unresolved or wider than a quarter of the span")
    rejection_reason = "; ".join(reasons) if reasons else None

    fig, ax = plt.subplots()
    ax.set_title("Double Hanger ResFlux Fit")
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





class ResonatorSpectroscopyVsFlux(ProtocolOperation):

    # --- analysis-only knobs for preprocessing ---
    MEDIAN_WINDOW = 16
    MAD_MULTIPLIER = 20.0
    # Must match the flux grid the CNN was trained on.
    N_CNN = 121
    N_START_TRIALS = 200
    MIN_CROP_POINTS = 10

    # Scale from this operation's native frequency units to GHz.  QICK reports
    # absolute RF MHz.  Downstream operations convert with this rather than
    # hardcoding a factor, so only the op that produced the numbers knows the
    # instrument convention (an OPX path would override it to 1e-9).
    FREQ_UNIT_TO_GHZ = 1e-3


    _SIM_QI = 8000.0
    _SIM_QE_MAG = 5000.0
    _SIM_THETA = 0.5
    _SIM_PHASE_OFF = 0.1
    _SIM_PHASE_SLOPE = 2e-9  # rad/Hz
    _SIM_TX_SLOPE = 5e-9  # /Hz
    _SIM_A = 1.0
    _SIM_PG = 0.9
    # Same units as the measured frequency axis (MHz for QICK).  The CNN's first
    # input channel is the raw absolute frequency in GHz, so a dummy curve at the
    # wrong scale trains/tests it on nonsense -- keep this near the real device.
    _SIM_FR = 6461.5          # mean dressed resonator frequency
    _SIM_CHI = 20.0           # amplitude of the flux modulation of f_r
    _SIM_GE_SPLITTING = 10.0  # separation of the |g> and |e> dips
    _SIM_NOISE_SIGMA = 0.2
    _SIM_EARTH_FLUX = 0.5     # flux offset the downstream model has to recover

    def __init__(self, params, set_flux_current=None):
        super().__init__()

        self.set_flux_current = set_flux_current

        self._register_inputs(
            repetitions=Repetition(params),
            steps=ResonatorSpecSteps(params),
            start_freq=StartReadoutFrequency(params),
            end_freq=EndReadoutFrequency(params),
            start_flux=StartFlux(params),
            end_flux=EndFlux(params),
            flux_steps=FluxSteps(params),
        )
        # No output parameters: this operation produces the measured f_r(flux)
        # curve as saved data.  The zero/half flux point is inferred downstream
        # by the offset-inverse ML model, which owns the fluxonium/resonator
        # theory and the parameter uncertainty that comes with it.

        self._register_correction_params(
            snr_threshold=ResSpecVsFluxSNRThreshold(params),
            min_good_fraction=ResSpecVsFluxMinGoodFraction(params),
        )

        self.condition = ("Success if enough flux points clear the SNR threshold "
                          "and a one-period curve was resampled for the CNN")
        self.independents = {"frequencies": [], "flux": []}
        self.dependents = {"signal": []}
        self.data_loc = None
        self.fr_fit = []
        self.fit_results = []
        self.snr = []
        self.linewidths = []
        self.rejection_reasons = []
        self.figure_paths = []

        # Filled by analyze(): the preprocessed curve handed to the CNN.
        self.good_mask = None
        self.flux_good = None
        self.fr_good = None
        self.period_est = None
        self.crop_window = None
        self.crop_span = None
        self.flux_crop = None
        self.fr_crop = None
        self.flux_resampled = None
        self.fr_resampled = None
        self.npz_path = None
        self.crop_error = None


    def _measure_qick(self) -> Path:
        logger.info("Starting qick resonator spectroscopy vs flux measurement")

        if self.set_flux_current is None:
            raise RuntimeError(
                "No flux current setter was supplied. Construct this operation "
                "with set_flux_current=<callable> so it can move the flux bias."
            )

        currents = np.linspace(
            self.start_flux(), self.end_flux(), int(self.flux_steps())
        )
        currents = np.round(currents / 0.125) * 0.125 # DMT is resolved to .125uA

        @pointer(independent("current", unit="uA"))
        def sweep_current():
            for value in currents:
                logger.debug(f"Setting flux current to {value} uA")
                self.set_flux_current(float(value))
                yield {"current": float(value)}

        # One full frequency sweep at each current set point.
        sweep = Sweep(sweep_current) @ FreqSweepProgram()
        logger.debug("Sweep created, running measurement")
        loc, da = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Measurement complete")

        return loc

    def _measure_dummy(self) -> Path:
        """
        Generate fake double-hanger res_spec-vs-flux data.
        Uses theoretical fr_g(flux) and fr_e(flux) to build a synthetic
        3D S21[flux, repetition, freq] map, with additive complex Gaussian noise.
        The interface matches _measure_quick: same inputs, same ddh5 layout, and measure the faked qubit with given scanned ranges.
        """

        logger.info("Starting dummy resonator spectroscopy vs flux (simulated fake data)")
        n_rep = self.repetitions()
        n_flux = self.flux_steps()
        n_freq = self.steps()
        start_flux = self.start_flux()
        end_flux = self.end_flux()
        start_freq = self.start_freq()
        end_freq = self.end_freq()
        freq = np.linspace(start_freq, end_freq, n_freq)
        flux_vals = np.linspace(start_flux, end_flux, n_flux)

        # Analytic stand-in for the flux dependence.  The fluxonium+resonator
        # diagonalisation that used to live here now belongs to the
        # offset-inverse model; this only has to be periodic in flux with
        # extrema at 0 and 1/2 to exercise the fitting and data-output path.
        # It is NOT a faithful fluxonium spectrum.
        flux_phase = 2 * np.pi * (flux_vals + self._SIM_EARTH_FLUX)
        fr_g_vs_flux = self._SIM_FR + self._SIM_CHI * np.cos(flux_phase)
        fr_e_vs_flux = fr_g_vs_flux + self._SIM_GE_SPLITTING

        def _dummy_double_hanger_signal():
            signals = np.empty((n_flux, n_freq), dtype=np.complex128)
            for i_flux in range(n_flux):
                noise = self._SIM_NOISE_SIGMA * (np.random.randn(n_freq) + 1j * np.random.randn(n_freq)) / np.sqrt(2.0)
                fr_g = fr_g_vs_flux[i_flux]
                fr_e = fr_e_vs_flux[i_flux]
                s21_clean = DoubleHangerResponseBruno.model(coordinates=freq, A=self._SIM_A, fr_g=fr_g, fr_e=fr_e, p_g=self._SIM_PG, Q_i=self._SIM_QI,
                                                            Q_e_mag=self._SIM_QE_MAG, theta=self._SIM_THETA, phase_offset=self._SIM_PHASE_OFF, phase_slope=self._SIM_PHASE_SLOPE, transmission_slope=self._SIM_TX_SLOPE)
                signals[i_flux, :] = s21_clean + noise
            return signals
        freq_grid = np.broadcast_to(freq[None, :], (n_flux, n_freq))
        flux_gird = np.broadcast_to(flux_vals[:, None], (n_flux, n_freq))
        sweep = (
            sweep_parameter("rep", range(n_rep))
            @ record_as(lambda:flux_gird, independent("current"))
            @ record_as(lambda: freq_grid, independent("frequency"))
            @ record_as(_dummy_double_hanger_signal, dependent("signal"))
        )
        loc, data = run_and_save_sweep(sweep, './data', 'my_data')
        logger.info(f"Dummy measurement complete, data saved to {loc}")
        return loc


    def _load_data_qick(self):
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)
        
        self.independents["flux"] = np.asarray(data["current"]["values"])
        self.independents["frequencies"] = np.asarray(data["freq"]["values"])[0, :]
        self.dependents["signal"] = np.asarray(data["signal"]["values"])

    def _load_data_dummy(self):
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        data = datadict_from_hdf5(path)

        # Same canonical layout, after averaging over the repetition axis.
        shape = (-1, int(self.flux_steps()), int(self.steps()))
        self.independents["flux"] = np.asarray(data["current"]["values"]).reshape(shape)[0, :, 0]
        self.independents["frequencies"] = np.asarray(data["frequency"]["values"]).reshape(shape)[0, 0, :]
        self.dependents["signal"] = np.asarray(data["signal"]["values"]).reshape(shape).mean(axis=0)


    def analyze(self):
        """Fit f_r at each flux point, filter, crop one flux period, resample.

        This reproduces the preprocessing in the offset-inverse model's
        "Generating experimental data for ML offset learning" notebook, so the
        curve saved here can be handed straight to the trained CNN.
        """
        # analyze() re-runs on every retry; these must not accumulate, and stale
        # crop results from a previous attempt must not be reported as current.
        self.fr_fit, self.snr, self.fit_results = [], [], []
        self.linewidths, self.rejection_reasons = [], []
        self.crop_error = None
        self.period_est = self.crop_window = self.npz_path = None
        self.crop_span = None
        self.flux_crop = self.fr_crop = None
        self.flux_resampled = self.fr_resampled = None

        with DatasetAnalysis(self.data_loc.parent, self.name) as ds:
            flux_axis = np.asarray(self.independents["flux"], dtype=float)
            freq_axis = np.asarray(self.independents["frequencies"], dtype=float)
            sig2d = np.asarray(self.dependents["signal"])

            # Each flux slice is an ordinary single-resonator spectroscopy trace,
            # so it is fitted by the same routine ResonatorSpectroscopy uses:
            # median-background SNR, bounded fit, linewidth and structural
            # rejection reasons all come from there.
            for i in range(sig2d.shape[0]):
                try:
                    # ret = ResonatorSpectroscopy.add_mag_and_unwind_and_fit(
                    #     freq_axis, sig2d[i, :]
                    # )
                    ret = add_mag_and_unwind_and_choose_fit(
                        freq_axis, sig2d[i, :]
                    )
                    plt.close(ret.fig)  # one figure per flux point would pile up
                    self.fit_results.append(ret.fit_result.params)
                    self.fr_fit.append(float(ret.fit_result.params["f_0"].value))
                    self.snr.append(float(ret.snr))
                    self.linewidths.append(float(ret.linewidth))
                    self.rejection_reasons.append(ret.rejection_reason)
                except Exception as exc:
                    # One bad flux point must not abort the whole sweep; it is
                    # dropped by the mask below.
                    logger.warning(f"Flux point {i} fit failed: {exc}")
                    self.fit_results.append(None)
                    self.fr_fit.append(np.nan)
                    self.snr.append(np.nan)
                    self.linewidths.append(np.nan)
                    self.rejection_reasons.append(str(exc))

            fr_fit = np.asarray(self.fr_fit, dtype=float)
            snr_arr = np.asarray(self.snr, dtype=float)
            linewidth_arr = np.asarray(self.linewidths, dtype=float)
            # A structurally rejected fit (f_0 at a sweep edge, unresolved
            # linewidth) is not usable even when its SNR is high.
            structurally_ok = np.asarray(
                [r is None for r in self.rejection_reasons], dtype=bool
            )

            good_mask, local_med, _dev = _robust_filter_fitted_curve(
                flux_axis=flux_axis,
                fr_fit=np.where(structurally_ok, fr_fit, np.nan),
                snr_list=snr_arr,
                freq_axis=freq_axis,
                snr_threshold=self.snr_threshold(),
                median_window=self.MEDIAN_WINDOW,
                mad_multiplier=self.MAD_MULTIPLIER,
            )
            self.good_mask = good_mask

            order = np.argsort(flux_axis[good_mask])
            flux_good = flux_axis[good_mask][order]
            fr_good = fr_fit[good_mask][order]
            self.flux_good, self.fr_good = flux_good, fr_good

            # Everything measured is recorded before the crop is attempted, so a
            # failure below still leaves a full record behind.
            ds.add(
                flux=flux_axis,
                fr_fit=fr_fit,
                snr=snr_arr,
                linewidth=linewidth_arr,
                structurally_ok=structurally_ok,
                good=good_mask,
                local_median=local_med,
            )

            # Raw flux sweep with the surviving fit on top.  Registered here,
            # ahead of the crop, so the report shows the measurement even when
            # the one-period selection cannot run.
            fig, ax = plt.subplots()
            mesh = ax.pcolormesh(flux_axis, freq_axis, np.abs(sig2d).T, shading="auto")
            fig.colorbar(mesh, ax=ax, label="|S| (a.u.)")
            ax.plot(flux_good, fr_good, "r.", ms=3, alpha=0.5, label="filtered fit")
            ax.set_xlabel("Flux / Current")
            ax.set_ylabel("Frequency")
            ax.set_title("Resonator response vs flux")
            ax.legend(fontsize="small")
            raw_path = ds._new_file_path(ds.savefolders[1], self.name, suffix="png")
            fig.savefig(raw_path)
            self.figure_paths.append(raw_path)

            # Crop exactly one flux period: infer the period, then slide a window
            # of that width and keep the cleanest one.  This needs well over one
            # period of well-fitted points, so it is the first thing to fail on a
            # short or noisy sweep -- treat that as a failed operation, not as a
            # crash that discards the report.
            try:
                (x_min, x_max, period_est,
                 starts, scores, lags, ac) = _auto_pick_one_period_window(
                    flux_good, fr_good,
                    n_points=self.N_CNN,
                    n_start_trials=self.N_START_TRIALS,
                )

                crop_mask = (flux_good >= x_min) & (flux_good <= x_max)
                flux_crop, fr_crop = flux_good[crop_mask], fr_good[crop_mask]
                if len(flux_crop) < self.MIN_CROP_POINTS:
                    raise ValueError(
                        f"Only {len(flux_crop)} fitted points inside the crop window "
                        f"[{x_min:.6g}, {x_max:.6g}]; need at least {self.MIN_CROP_POINTS}."
                    )

                flux_resampled, fr_resampled = _resample_one_period(
                    flux_crop, fr_crop, n_points=self.N_CNN
                )
            except Exception as exc:
                self.crop_error = str(exc)
                logger.warning(
                    f"Could not crop one flux period ({exc}); the raw sweep and "
                    f"fitted curve are still saved, but there is no CNN input."
                )
                return

            self.period_est = float(period_est)
            self.crop_window = (x_min, x_max)
            self.flux_crop, self.fr_crop = flux_crop, fr_crop
            self.flux_resampled, self.fr_resampled = flux_resampled, fr_resampled
            # Span actually covered by the resampled grid.  _resample_one_period
            # spans the nearest *measured* points inside the window, so this is
            # shorter than x_max - x_min by up to one sample gap.  Downstream
            # offset->current conversion must use this, not the window width.
            self.crop_span = float(flux_crop.max() - flux_crop.min())

            ds.add(
                crop_x_min=float(x_min),
                crop_x_max=float(x_max),
                period_estimate=float(period_est),
                flux_resampled=flux_resampled,
                fr_resampled=fr_resampled,
            )

            # Companion .npz for the CNN.  Keys are named (not positional) so
            # that the ML notebook's data["flux_resampled"] / data["fr_resampled"]
            # lookups resolve.
            self.npz_path = ds._new_file_path(
                ds.savefolders[1], f"{self.name}_for_cnn", suffix="npz"
            )
            np.savez_compressed(
                self.npz_path,
                flux_axis=flux_axis,
                freq_axis=freq_axis,
                signal2d=sig2d,
                flux_good=flux_good,
                fr_good=fr_good,
                x_min=x_min,
                x_max=x_max,
                period_est=period_est,
                flux_crop=flux_crop,
                fr_crop=fr_crop,
                flux_resampled=flux_resampled,
                fr_resampled=fr_resampled,
            )
            logger.info(f"Saved CNN input curve to {self.npz_path}")

            fig, ax = plt.subplots()
            mesh = ax.pcolormesh(flux_axis, freq_axis, np.abs(sig2d).T, shading="auto")
            fig.colorbar(mesh, ax=ax, label="|S| (a.u.)")
            ax.plot(flux_good, fr_good, "r.", ms=3, alpha=0.5, label="filtered fit")
            ax.plot(flux_resampled, fr_resampled, "wo", ms=3, mec="k", mew=0.5,
                    label=f"resampled {self.N_CNN} pts")
            ax.axvline(x_min, color="cyan", linestyle="--", label="one-period crop")
            ax.axvline(x_max, color="cyan", linestyle="--")
            ax.set_xlabel("Flux / Current")
            ax.set_ylabel("Frequency")
            ax.set_title("Resonator response vs flux, cropped to one period")
            ax.legend(fontsize="small")
            image_path = ds._new_file_path(
                ds.savefolders[1], f"{self.name}_one_period", suffix="png"
            )
            fig.savefig(image_path)
            self.figure_paths.append(image_path)

            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))
            ax1.plot(lags, ac, "-")
            ax1.axvline(period_est, color="red", linestyle="--", label="estimated period")
            ax1.set_xlabel("Lag in flux / current")
            ax1.set_ylabel("Autocorrelation")
            ax1.legend(fontsize="small")
            ax2.plot(starts, scores, "o-", ms=3)
            ax2.axvline(x_min, color="red", linestyle="--", label="chosen start")
            ax2.set_xlabel("Window start")
            ax2.set_ylabel("Window score")
            ax2.set_yscale("log")
            ax2.legend(fontsize="small")
            fig2.suptitle("One-period selection diagnostics")
            fig2.tight_layout()
            diag_path = ds._new_file_path(
                ds.savefolders[1], f"{self.name}_period_selection", suffix="png"
            )
            fig2.savefig(diag_path)
            self.figure_paths.append(diag_path)


    def evaluate(self) -> OperationStatus:
        """
        Final evaluation: is the measured f_r(flux) curve good enough to hand to
        the offset-inverse model?  Criterion: enough flux points fitted with SNR
        above the SNR threshold.  Whether the curve actually locates a zero/half
        flux point is decided downstream, not here.
        """

        if not self.snr or len(self.snr) == 0:
            self.report_output.append("No SNR computed. Did analyze() run?")
            logger.warning("No SNR computed. Did analyze() run?")
            return EvaluateResult(
                OperationStatus.FAILURE,
                [CheckResult("snr_quality", False, "analyze() produced no fits")],
            )
        threshold = self.snr_threshold()
        min_fraction = self.min_good_fraction()
        snr_arr = np.asarray(self.snr, dtype=float)
        good_fraction = float(np.nanmean(snr_arr >= threshold))
        all_snr_good = good_fraction >= min_fraction

        curve_ok = (
            self.fr_resampled is not None
            and len(self.fr_resampled) == self.N_CNN
            and np.all(np.isfinite(self.fr_resampled))
        )

        x_min, x_max = self.crop_window if self.crop_window else (float("nan"),) * 2
        msg = (
            "## Resonator Spectroscopy vs Flux\n"
            f"Flux points (traces): {len(self.snr)}\n"
            f"Structurally valid fits: "
            f"{sum(r is None for r in self.rejection_reasons)}/{len(self.rejection_reasons)}\n"
            f"SNR min/median/max: {np.nanmin(snr_arr):.2f} / "
            f"{np.nanmedian(snr_arr):.2f} / {np.nanmax(snr_arr):.2f}\n"
            f"Pass threshold (per trace): SNR >= {threshold}\n"
            f"Required good fraction: {min_fraction:.0%}\n"
            f"Good-SNR fraction: {good_fraction:.2%}\n"
            f"Points kept after filtering: "
            f"{0 if self.good_mask is None else int(np.sum(self.good_mask))}\n"
            f"Inferred period: {self.period_est}\n"
            f"One-period crop: [{x_min:.6g}, {x_max:.6g}]\n"
        )
        if curve_ok:
            msg += f"Resampled curve for CNN: {self.N_CNN} points -> `{self.npz_path}`\n"
        else:
            msg += f"**No CNN input produced:** {self.crop_error}\n"
        # Append, never assign: execute() puts the "### ATTEMPT n" marker in here
        # before measure(), and correct() adds figures after, so overwriting
        # would discard the previous attempt's whole section on a retry.
        self.report_output.append(msg)

        checks = [
            CheckResult(
                "snr_quality", all_snr_good,
                f"{good_fraction:.0%} of {len(self.snr)} flux points have SNR >= "
                f"{threshold} (need {min_fraction:.0%})",
            ),
            CheckResult(
                "cnn_input", curve_ok,
                f"{self.N_CNN}-point one-period curve saved" if curve_ok
                else f"no one-period curve: {self.crop_error}",
            ),
        ]
        status = (OperationStatus.SUCCESS if all(c.passed for c in checks)
                  else OperationStatus.FAILURE)
        for check in checks:
            if not check.passed:
                logger.warning(f"{check.name}: {check.description}")
        return EvaluateResult(status, checks)

    def correct(self, result: EvaluateResult) -> EvaluateResult:
        """Put every figure in the report, each with a caption.

        The base implementation appends only ``figure_paths[-1]``, so as soon as
        the crop succeeds and later plots exist, the raw flux sweep and the crop
        overlay are dropped -- and the raw sweep is the one worth looking at.
        """
        captions = ["**Flux sweep:**", "**One-period crop:**", "**Period selection:**"]
        figures = list(self.figure_paths)
        # Cleared so the base class does not append the last figure a second time.
        self.figure_paths.clear()
        for i, path in enumerate(figures):
            caption = captions[i] if i < len(captions) else f"**Figure {i + 1}:**"
            self.report_output.extend([f"\n{caption}\n", path.resolve()])

        return super().correct(result)

