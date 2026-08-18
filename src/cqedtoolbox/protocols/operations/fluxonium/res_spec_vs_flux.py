import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths
from scipy.sparse import diags, kron, identity, csc_matrix
from scipy.sparse.linalg import eigsh

plt.switch_backend("agg")

from labcore.data.datadict_storage import datadict_from_hdf5
from labcore.analysis import DatasetAnalysis
from labcore.analysis.fit import Fit
from labcore.measurement.sweep import sweep_parameter, Sweep, pointer
from labcore.measurement.storage import run_and_save_sweep
from labcore.measurement.record import record_as, independent, dependent

from labcore.protocols.base import ProtocolOperation, OperationStatus
# from cqedtoolbox.protocols.parameters import (
#     Repetition, ResonatorSpecSteps, StartReadoutFrequency, EndReadoutFrequency,
#     StartFlux, EndFlux, FluxSteps,
# )
from parameters import (
    Repetition, ResonatorSpecSteps, StartReadoutFrequency, EndReadoutFrequency,
    StartFlux, EndFlux, FluxSteps,
)
from cqedtoolbox.measurement_lib.qick.single_transmon_v2 import FreqSweepProgram

logger = logging.getLogger(__name__)

#----------- THIS IS ALL DUMMY DATA GENERATION CODE, NOT USED IN REAL MEASUREMENTS ----------------

class DoubleHangerResponseBruno(Fit):
    """Shared double-hanger response model for simulated flux sweeps."""

    @staticmethod
    def model(
        coordinates: np.ndarray, A: float, fr_g: float, fr_e: float,
        p_g: float, Q_i: float, Q_e_mag: float, theta: float,
        phase_offset: float, phase_slope: float, transmission_slope: float,
    ):
        x = coordinates
        amp_correction_g = A * (1 + transmission_slope * (x - fr_g) / fr_g)
        amp_correction_e = A * (1 + transmission_slope * (x - fr_e) / fr_e)
        phase_correction = np.exp(1j * (phase_slope * x + phase_offset))

        if Q_e_mag == 0:
            Q_e_mag = 1e-12
        Q_e_complex = Q_e_mag * np.exp(-1j * theta)
        Q_c = 1.0 / (1.0 / Q_e_complex).real
        Q_l = 1.0 / (1.0 / Q_c + 1.0 / Q_i)
        response_g = 1 - Q_l / np.abs(Q_e_mag) * np.exp(1j * theta) / (
            1.0 + 2j * Q_l * (x - fr_g) / fr_g
        )
        response_e = 1 - Q_l / np.abs(Q_e_mag) * np.exp(1j * theta) / (
            1.0 + 2j * Q_l * (x - fr_e) / fr_e
        )
        return (
            p_g * response_g * amp_correction_g
            + (1 - p_g) * response_e * amp_correction_e
        ) * phase_correction

def _fluxonium_basis(EC, EL, EJ, flux_ext, levels=8, grid=2000, flux_max=12*np.pi):
    """Creat Fluxonium Hamiltonian, return (E, n_Q projected into eigenbasis) using FD in phase."""

    flux = np.linspace(-flux_max, flux_max, grid)
    dflux = flux[1] - flux[0]
    main, off = 2.0*np.ones(grid), -1.0*np.ones(grid-1)
    lap = diags([off, main, off], offsets=[-1, 0, 1]) / (dflux**2)
    T = -4.0 * EC * lap
    V = 0.5*EL*(flux**2) - EJ*np.cos(flux-flux_ext)
    H = T + diags(V, 0)

    evals, evecs = eigsh(H, k=levels, which="SA")
    idx = np.argsort(evals)
    E = np.array(evals[idx])
    Psi = np.array(evecs[:, idx])

    offp =  np.ones(grid-1) / (2.0 * dflux)
    offm = -np.ones(grid-1) / (2.0 * dflux)
    d1 = diags([offm, np.zeros(grid), offp], [-1, 0, 1], dtype=np.complex128)
    n_grid = (-1j) * d1
    nQ = Psi.conj().T @ (n_grid @ Psi)
    return E, nQ


def _resonator_ops_from_fr(fr_bare, N_phot=6):
    """Creat Resonator Hamiltonian, return H_res (GHz), n_R = i(a†-a), N = a†a in N_phot Fock basis."""

    a = np.zeros((N_phot, N_phot), dtype=complex)
    for n in range(1, N_phot):
        a[n-1, n] = np.sqrt(n)
    adag = a.conj().T
    Hres = fr_bare * (adag @ a + 0.5 * np.eye(N_phot))
    nR = 1j * (adag - a)
    Nph = adag @ a
    return csc_matrix(Hres), csc_matrix(nR), csc_matrix(Nph)


def _build_total_H(EC, EL, EJ, flux_ext, fr_bare, g, q_levels=8, N_phot=6, grid=2000, flux_max=12*np.pi):
    EQ, nQ = _fluxonium_basis(EC, EL, EJ, flux_ext, levels=q_levels, grid=grid, flux_max=flux_max)
    Hq = diags(EQ, 0, format='csc')
    Iq = identity(q_levels, format='csc')
    Hres, nR, Nph = _resonator_ops_from_fr(fr_bare, N_phot=N_phot)
    Ir = identity(N_phot, format='csc')
    Hint = g * kron(csc_matrix(nQ), nR)
    Htot = kron(Hq, Ir) + kron(Iq, Hres) + Hint
    return Htot, Nph, q_levels, N_phot


def _pick_state(E, V, Pg_full, Pe_full, Nph_full, manifold, n_target):
    """Select the eigenstate closest to |manifold, n_target>, since they might not be lowest 4 eigenstates.
    For g/e state of qubit, expection value of P_g/P_e projector should close to 1. For 0/1 state of resonator expectation of N operator should close to 0/1"""

    P = Pg_full if manifold == "g" else Pe_full
    best_E = None
    best_score = np.inf
    for j in range(E.size):
        v = V[:, j]
        p = float(np.real_if_close(v.conj().T @ (P @ v)))
        nbar = float(np.real_if_close(v.conj().T @ (Nph_full @ v)))
        score = (1.0 - p) + abs(nbar - n_target)
        if score < best_score:
            best_score = score
            best_E = E[j]
    return best_E


def _readout_frequencies(EC, EL, EJ, flux_ext, fr, g, q_levels=10, N_phot=6, grid=2000, flux_max=12*np.pi):
    """Simulate frequencies fr_g and fr_e of specific flux point"""

    Htot, Nph, Lq, Nr = _build_total_H(EC, EL, EJ, flux_ext, fr, g, q_levels=q_levels,
        N_phot=N_phot, grid=grid, flux_max=flux_max)
    k_eval = min(4 * Lq, Htot.shape[0] - 2) # Make sure to include g0, g1, e0, e1
    evals, evecs = eigsh(Htot, k=k_eval, which="SA")
    idx = np.argsort(evals)
    E = evals[idx]
    V = evecs[:, idx]

    Pg = np.zeros((Lq, Lq))
    Pg[0, 0] = 1.0
    Pe = np.zeros((Lq, Lq))
    Pe[1, 1] = 1.0
    Pg_full = kron(csc_matrix(Pg), identity(Nr, format="csc"))
    Pe_full = kron(csc_matrix(Pe), identity(Nr, format="csc"))
    Nph_full = kron(identity(Lq, format="csc"), Nph)

    Eg0 = _pick_state(E, V, Pg_full, Pe_full, Nph_full, manifold="g", n_target=0)
    Eg1 = _pick_state(E, V, Pg_full, Pe_full, Nph_full, manifold="g", n_target=1)
    Ee0 = _pick_state(E, V, Pg_full, Pe_full, Nph_full, manifold="e", n_target=0)
    Ee1 = _pick_state(E, V, Pg_full, Pe_full, Nph_full, manifold="e", n_target=1)
    fr_g = Eg1 - Eg0
    fr_e = Ee1 - Ee0
    return fr_g, fr_e

#--------------------------------------------------------------------------------------------------

def fit_lorentzian_notches(
    frequency, magnitude, *, median_window=21, smooth_sigma=1.5,
    min_prominence=0.15, min_snr=1.5, bic_threshold=6.0,
    min_bic_improvement=10.0,
):
    """Return one or two statistically significant Lorentzian dip centres.

    A median filter removes the broad background before candidate dips are
    detected. The final Lorentzian fit is made against the unsmoothed residual,
    so ``fit`` is directly comparable with the measured magnitude.
    """
    frequency = np.asarray(frequency, float)
    magnitude = np.asarray(magnitude, float)
    valid = np.isfinite(frequency) & np.isfinite(magnitude)
    frequency, magnitude = frequency[valid], magnitude[valid]
    order = np.argsort(frequency)
    frequency, magnitude = frequency[order], magnitude[order]

    if len(frequency) < 7 or np.any(np.diff(frequency) <= 0):
        raise ValueError("frequency must contain at least 7 distinct points")
    if median_window < 3 or median_window > len(frequency) or median_window % 2 == 0:
        raise ValueError(
            "median_window must be odd, at least 3, and no longer than the trace"
        )

    background = median_filter(magnitude, size=median_window, mode="nearest")
    trace = magnitude - background
    dip = -gaussian_filter1d(trace, smooth_sigma)
    peaks, properties = find_peaks(
        dip, prominence=min_prominence * np.ptp(trace), distance=3
    )
    if len(peaks) == 0:
        raise ValueError("No resonant dip")
    peaks = peaks[np.argsort(properties["prominences"])[::-1][:2]]
    frequency_step = np.median(np.diff(frequency))

    def model(frequency_values, offset, *components):
        result = np.full_like(frequency_values, offset, dtype=float)
        for amplitude, centre, hwhm in np.reshape(components, (-1, 3)):
            result -= amplitude / (1 + ((frequency_values - centre) / hwhm) ** 2)
        return result

    def fit(indices):
        widths = peak_widths(dip, indices, rel_height=0.5)[0]
        parameters = [np.median(trace)]
        for index, width in zip(indices, widths):
            hwhm = np.clip(
                width * frequency_step / 2,
                frequency_step,
                (frequency[-1] - frequency[0]) / 5,
            )
            parameters.extend([dip[index], frequency[index], hwhm])

        n_notches = len(indices)
        lower = [-np.inf] + n_notches * [0, frequency[0], frequency_step / 2]
        upper = [np.inf] + n_notches * [
            np.inf,
            frequency[-1],
            (frequency[-1] - frequency[0]) / 4,
        ]
        parameters, _ = curve_fit(
            model,
            frequency,
            trace,
            p0=parameters,
            bounds=(lower, upper),
            maxfev=10_000,
        )
        residuals = trace - model(frequency, *parameters)
        mean_square_residual = max(
            float(np.mean(residuals**2)), np.finfo(float).tiny
        )
        bic = (
            len(trace) * np.log(mean_square_residual)
            + len(parameters) * np.log(len(trace))
        )
        noise = np.std(residuals)
        snr = np.inf if noise == 0 else (
            np.abs(np.reshape(parameters[1:], (-1, 3))[:, 0]) / (4 * noise)
        )
        return parameters, bic, snr

    flat_residuals = trace - np.mean(trace)
    flat_bic = (
        len(trace)
        * np.log(max(float(np.mean(flat_residuals**2)), np.finfo(float).tiny))
        + np.log(len(trace))
    )

    single, single_bic, single_snr = min(
        (fit([peak]) for peak in peaks), key=lambda result: result[1]
    )
    parameters, selected_bic, selected_snr = single, single_bic, single_snr

    if len(peaks) == 2:
        double, double_bic, double_snr = fit(peaks)
        if double_bic <= single_bic - bic_threshold and np.all(double_snr >= min_snr):
            parameters, selected_bic, selected_snr = double, double_bic, double_snr

    if flat_bic - selected_bic < min_bic_improvement or np.any(selected_snr < min_snr):
        raise ValueError("No statistically significant resonance")

    components = np.reshape(parameters[1:], (-1, 3))
    components = components[np.argsort(components[:, 1])]
    return {
        "n_notches": len(components),
        "centres": components[:, 1],
        "frequency": frequency,
        "fit": background + model(frequency, *parameters),
    }

class ResonatorSpectroscopyVsFlux(ProtocolOperation):

    MIN_FITTED_FRACTION = 0.8

    # True fake data params (may change to any)
    _SIM_QI = 8000.0
    _SIM_QE_MAG = 5000.0
    _SIM_THETA = 0.5
    _SIM_PHASE_OFF = 0.1
    _SIM_PHASE_SLOPE = 2e-9  # rad/Hz
    _SIM_TX_SLOPE = 5e-9  # /Hz
    _SIM_A = 1.0
    _SIM_PG = 0.9
    _SIM_EC = 1.0  # followings are params for fluxonium
    _SIM_EL = 0.5
    _SIM_EJ = 5.3
    _SIM_FR = 4.07
    _SIM_G = 0.067
    _SIM_NOISE_SIGMA = 0.2
    _SIM_EARTH_FLUX = 0.5

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

        self.condition = (
            "Success if Lorentzian notches are found in most flux traces"
        )
        self.independents = {"frequencies": [], "flux": []}
        self.dependents = {"signal": []}
        self.data_loc = None
        self.figure_paths = []
        self.notch_centres = None
        self.notch_counts = None
        self.fit_magnitude = None


    def _measure_dummy(self) -> Path:
        """Generate dummy data with the same flux-frequency layout as QICK."""

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

        fr_g_vs_flux=[]
        fr_e_vs_flux=[]
        for flux_ext in flux_vals:
            fr_g, fr_e = _readout_frequencies(self._SIM_EC, self._SIM_EL, self._SIM_EJ, flux_ext + self._SIM_EARTH_FLUX, self._SIM_FR, self._SIM_G, q_levels=10, N_phot=5)
            fr_g_vs_flux.append(fr_g)
            fr_e_vs_flux.append(fr_e)
        fr_g_vs_flux = np.asarray(fr_g_vs_flux)
        fr_e_vs_flux = np.asarray(fr_e_vs_flux)

        def _dummy_double_hanger_signal():
            signals = np.empty((n_flux, n_freq), dtype=np.complex128)
            for i_flux in range(n_flux):
                noise = self._SIM_NOISE_SIGMA * (np.random.randn(n_freq) + 1j * np.random.randn(n_freq)) / np.sqrt(2.0)
                fr_g = fr_g_vs_flux[i_flux]
                fr_e = fr_e_vs_flux[i_flux]
                s21_clean = DoubleHangerResponseBruno.model(
                    coordinates=freq,
                    A=self._SIM_A,
                    fr_g=fr_g,
                    fr_e=fr_e,
                    p_g=self._SIM_PG,
                    Q_i=self._SIM_QI,
                    Q_e_mag=self._SIM_QE_MAG,
                    theta=self._SIM_THETA,
                    phase_offset=self._SIM_PHASE_OFF,
                    phase_slope=self._SIM_PHASE_SLOPE,
                    transmission_slope=self._SIM_TX_SLOPE,
                )
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
        loc, _ = run_and_save_sweep(sweep, "data", self.name)
        logger.info(f"Dummy measurement complete, data saved to {loc}")
        return loc


    def _load_data(self, frequency_key):
        """Load DDH5 data as flux, frequency, and signal[flux, frequency]."""
        path = self.data_loc / "data.ddh5"
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")

        data = datadict_from_hdf5(path)
        shape = int(self.flux_steps()), int(self.steps())
        signal = np.asarray(data["signal"]["values"]).reshape((-1, *shape)).mean(0)

        def axis(name, index):
            values = np.asarray(data[name]["values"])
            if values.size == shape[index]:
                return values.reshape(-1)
            grid = values.reshape((-1, *shape))[0]
            return grid[:, 0] if index == 0 else grid[0]

        flux = axis("current", 0)
        frequencies = axis(frequency_key, 1)
        if not (
            np.isfinite(flux).all()
            and np.isfinite(frequencies).all()
            and np.isfinite(signal).all()
        ):
            raise ValueError("Sweep data contains non-finite values")

        flux_order, frequency_order = np.argsort(flux), np.argsort(frequencies)
        self.independents["flux"] = flux[flux_order]
        self.independents["frequencies"] = frequencies[frequency_order]
        self.dependents["signal"] = signal[flux_order][:, frequency_order]

    def _load_data_dummy(self):
        self._load_data("frequency")


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
        currents = np.round(currents / 0.125) * 0.125  # DMT resolution: 0.125 uA

        @pointer(independent("current", unit="uA"))
        def sweep_current():
            for value in currents:
                logger.debug(f"Setting flux current to {value} uA")
                self.set_flux_current(float(value))
                yield {"current": float(value)}

        sweep = Sweep(sweep_current) @ FreqSweepProgram()
        logger.debug("Sweep created, running measurement")
        loc, _ = run_and_save_sweep(sweep, "data", self.name)
        logger.info("Measurement complete")

        return loc

    def _load_data_qick(self):
        self._load_data("freq")


    def _analyze_default(self):
        flux = self.independents["flux"]
        frequencies = self.independents["frequencies"]
        magnitude = np.abs(self.dependents["signal"])
        self.figure_paths.clear()
        self.notch_centres = np.full((len(flux), 2), np.nan)
        self.notch_counts = np.zeros(len(flux), dtype=int)
        self.fit_magnitude = np.full_like(magnitude, np.nan, dtype=float)

        for index, trace in enumerate(magnitude):
            try:
                result = fit_lorentzian_notches(frequencies, trace)
            except (RuntimeError, ValueError) as error:
                logger.info("No notch fit at flux index %d: %s", index, error)
                continue

            centres = result["centres"]
            self.notch_centres[index, : len(centres)] = centres
            self.notch_counts[index] = len(centres)
            self.fit_magnitude[index] = result["fit"]

        with DatasetAnalysis(self.data_loc.parent, self.name) as ds:
            ds.add(
                flux=flux,
                frequencies=frequencies,
                signal_magnitude=magnitude,
                fitted_magnitude=self.fit_magnitude,
                notch_centres=self.notch_centres,
                notch_counts=self.notch_counts,
            )

            figure, (map_axis, trace_axis) = plt.subplots(
                1, 2, figsize=(13, 5), constrained_layout=True
            )
            mesh = map_axis.pcolormesh(
                flux, frequencies, magnitude.T, shading="auto", cmap="inferno"
            )
            figure.colorbar(mesh, ax=map_axis, label="|S| (a.u.)")
            flux_grid = np.broadcast_to(np.asarray(flux)[:, None], self.notch_centres.shape)
            fitted = np.isfinite(self.notch_centres)
            map_axis.plot(flux_grid[fitted], self.notch_centres[fitted], "w.", ms=5)
            map_axis.set(
                xlabel="Flux current (uA)",
                ylabel="Readout frequency",
                title="Resonator response with fitted notch centres",
            )

            offset = max(
                float(np.nanmax(magnitude) - np.nanmin(magnitude)), np.finfo(float).eps
            ) * 1.2
            for index, (trace, fit) in enumerate(zip(magnitude, self.fit_magnitude)):
                trace_axis.plot(frequencies, trace + index * offset, "k-", lw=0.5)
                trace_axis.plot(frequencies, fit + index * offset, "r--", lw=0.7)
            trace_axis.plot([], [], "k-", label="measured")
            trace_axis.plot([], [], "r--", label="Lorentzian fit")
            trace_axis.set(
                xlabel="Readout frequency",
                ylabel="Magnitude + trace offset",
                title="Trace fits",
            )
            trace_axis.legend(fontsize="small")

            image_path = ds._new_file_path(ds.savefolders[1], self.name, suffix="png")
            figure.savefig(image_path)
            plt.close(figure)
            self.figure_paths.append(image_path)


    def evaluate(self) -> OperationStatus:
        """Evaluate whether the notch fit succeeded for enough raw traces."""
        if self.notch_counts is None:
            self.report_output = ["No Lorentzian-notch fits computed. Did analyze() run?"]
            logger.warning("No Lorentzian-notch fits computed. Did analyze() run?")
            return OperationStatus.FAILURE

        notch_counts = self.notch_counts
        fitted_fraction = float(np.mean(notch_counts > 0))
        one_notch = int(np.sum(notch_counts == 1))
        two_notches = int(np.sum(notch_counts == 2))
        successful = fitted_fraction >= self.MIN_FITTED_FRACTION
        message = (
            "## Resonator Spectroscopy vs Flux\n"
            f"Flux points (traces): {len(notch_counts)}\n"
            f"Fitted traces: {int(np.sum(notch_counts > 0))}/{len(notch_counts)} "
            f"({fitted_fraction:.2%})\n"
            f"One-notch fits: {one_notch}\n"
            f"Two-notch fits: {two_notches}\n"
            f"Required fitted fraction: {self.MIN_FITTED_FRACTION:.0%}\n"
        )
        self.report_output = [message]
        if not successful:
            logger.warning(
                "Only %.2f%% of flux traces had a significant Lorentzian notch",
                fitted_fraction * 100,
            )
        return OperationStatus.SUCCESS if successful else OperationStatus.FAILURE
