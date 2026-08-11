"""Fit the measured resonator-vs-flux curve to fluxonium+resonator theory.

Analysis-only: it reuses the curve ResonatorSpectroscopyVsFlux already extracted
and the flux calibration FluxOffsetInference already produced, and fits the
coupling g and the bare resonator frequency fr with EJ/EC/EL held fixed.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import lmfit
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis
from labcore.protocols.base import (ProtocolOperation, OperationStatus,
                                    CorrectionParameter, CheckResult, EvaluateResult)
from parameters import (
    ECParam, ELParam, EJParam, ZeroFluxCurrent, HalfFluxCurrent,
    CouplingG, ResonatorFr,
)
from cqedtoolbox.fitfuncs.fluxonium_spectrum import FluxoniumResonatorFit

logger = logging.getLogger(__name__)


@dataclass
class TheoryFitMaxRms(CorrectionParameter):
    name: str = field(default="fluxonium_theory_fit_max_rms_mhz", init=False)
    description: str = field(
        default="Largest acceptable RMS fit residual, MHz", init=False
    )

    def _qick_getter(self):
        return self.params.corrections.fluxonium_theory_fit.max_rms_mhz()

    def _qick_setter(self, v):
        self.params.corrections.fluxonium_theory_fit.max_rms_mhz(v)

    def _opx_getter(self):
        return self.params.corrections.fluxonium_theory_fit.max_rms_mhz()

    def _opx_setter(self, v):
        self.params.corrections.fluxonium_theory_fit.max_rms_mhz(v)


class FluxoniumResonatorTheoryFit(ProtocolOperation):

    # Each model evaluation diagonalizes a joint Hilbert space per flux point,
    # and lmfit needs tens of evaluations, so the curve is downsampled. Two free
    # parameters do not need more than this; the notebook's known-good fit used
    # about 21 points.
    N_THEORY = 31
    # Physically plausible coupling range, GHz. Bounds matter because labcore's
    # Fit only seeds a value from guess() and would otherwise let the optimizer
    # wander into nonsense.
    G_BOUNDS_GHZ = (1e-3, 0.5)
    FR_SEARCH_GHZ = 0.5

    def __init__(self, params, source=None):
        super().__init__()

        self.source = source

        self._register_inputs(
            EC=ECParam(params),
            EL=ELParam(params),
            EJ=EJParam(params),
            zero_flux_current=ZeroFluxCurrent(params),
            half_flux_current=HalfFluxCurrent(params),
        )
        self._register_outputs(
            coupling_g=CouplingG(params),
            resonator_fr=ResonatorFr(params),
        )
        self._register_correction_params(
            max_rms_mhz=TheoryFitMaxRms(params),
        )

        self._register_success_update(self.coupling_g, lambda: float(self.g_fit))
        self._register_success_update(self.resonator_fr, lambda: float(self.fr_fit))

        self.condition = "Success if the theory fit converges with a small residual"
        self.max_attempts = 3

        self.fit_result = None
        self.flux_phi0 = None
        self.fr_measured_ghz = None
        self.fr_model_ghz = None
        self.g_fit = None
        self.fr_fit = None
        self.rms_mhz = None
        self.fit_error = None

    # --- no measurement of its own -------------------------------------------

    def _measure_qick(self) -> Path:
        return self._reuse_source_data()

    def _measure_dummy(self) -> Path:
        return self._reuse_source_data()

    def _reuse_source_data(self) -> Path:
        if self.source is None or self.source.data_loc is None:
            raise RuntimeError(
                "FluxoniumResonatorTheoryFit needs the ResonatorSpectroscopyVsFlux "
                "operation it follows; pass source=<that operation>."
            )
        return self.source.data_loc

    def _load_data_qick(self):
        pass

    def _load_data_dummy(self):
        pass

    # --- analysis -------------------------------------------------------------

    def analyze(self):
        self.fit_result = None
        self.g_fit = self.fr_fit = self.rms_mhz = None
        self.flux_phi0 = self.fr_measured_ghz = self.fr_model_ghz = None
        self.fit_error = None

        flux_good = getattr(self.source, "flux_good", None)
        fr_good = getattr(self.source, "fr_good", None)
        if flux_good is None or fr_good is None or len(flux_good) < 4:
            self.fit_error = "source operation produced no filtered f_r(flux) curve"
            logger.warning(self.fit_error)
            return

        zero = float(self.zero_flux_current())
        half = float(self.half_flux_current())
        if not np.isfinite(zero) or not np.isfinite(half) or half == zero:
            self.fit_error = (
                f"flux calibration is unusable (zero={zero}, half={half}); "
                f"the flux axis cannot be built"
            )
            logger.warning(self.fit_error)
            return

        # Current -> external flux in units of the flux quantum. Half a period
        # of current spans half a flux quantum, hence the factor of two.
        flux_all = (np.asarray(flux_good, dtype=float) - zero) / (half - zero) / 2
        fr_all = np.asarray(fr_good, dtype=float) * self.source.FREQ_UNIT_TO_GHZ

        # Evenly spaced subset, so the fit sees the whole flux range rather than
        # a dense patch of it.
        if len(flux_all) > self.N_THEORY:
            keep = np.unique(np.linspace(0, len(flux_all) - 1, self.N_THEORY).astype(int))
        else:
            keep = np.arange(len(flux_all))
        self.flux_phi0 = flux_all[keep]
        self.fr_measured_ghz = fr_all[keep]

        fit = FluxoniumResonatorFit(self.flux_phi0, self.fr_measured_ghz)
        fit.EC, fit.EL, fit.EJ = float(self.EC()), float(self.EL()), float(self.EJ())

        fr_seed = float(np.mean(self.fr_measured_ghz))
        fit_params = {
            "g": lmfit.Parameter("g", value=60e-3,
                                 min=self.G_BOUNDS_GHZ[0], max=self.G_BOUNDS_GHZ[1]),
            "fr": lmfit.Parameter("fr", value=fr_seed,
                                  min=fr_seed - self.FR_SEARCH_GHZ,
                                  max=fr_seed + self.FR_SEARCH_GHZ),
        }

        logger.info(
            f"Fitting {len(self.flux_phi0)} flux points with "
            f"EC={fit.EC}, EL={fit.EL}, EJ={fit.EJ} (fixed)"
        )
        try:
            self.fit_result = fit.run(params=fit_params)
        except Exception as exc:
            self.fit_error = f"theory fit failed: {exc}"
            logger.warning(self.fit_error)
            return

        self.g_fit = float(self.fit_result.params["g"].value)
        self.fr_fit = float(self.fit_result.params["fr"].value)
        self.fr_model_ghz = np.asarray(self.fit_result.eval(), dtype=float)
        self.rms_mhz = float(
            np.sqrt(np.mean((self.fr_model_ghz - self.fr_measured_ghz) ** 2)) * 1e3
        )
        logger.info(
            f"g = {self.g_fit * 1e3:.3f} MHz, fr = {self.fr_fit:.6f} GHz, "
            f"RMS residual = {self.rms_mhz:.3f} MHz"
        )

        with DatasetAnalysis(self.data_loc.parent, self.name) as ds:
            ds.add(
                flux_phi0=self.flux_phi0,
                fr_measured_ghz=self.fr_measured_ghz,
                fr_model_ghz=self.fr_model_ghz,
                g_ghz=self.g_fit,
                fr_ghz=self.fr_fit,
                rms_residual_mhz=self.rms_mhz,
            )

            fig, (ax, ax_res) = plt.subplots(
                2, 1, sharex=True, figsize=(7, 6),
                gridspec_kw={"height_ratios": [3, 1]},
            )
            ax.plot(self.flux_phi0, self.fr_measured_ghz, "o", ms=4, label="measured")
            ax.plot(self.flux_phi0, self.fr_model_ghz, "-", label="theory fit")
            ax.set_ylabel("Resonator frequency (GHz)")
            ax.set_title(
                f"g = {self.g_fit * 1e3:.2f} MHz,  fr = {self.fr_fit:.5f} GHz"
            )
            ax.legend(fontsize="small")
            ax_res.plot(self.flux_phi0,
                        (self.fr_model_ghz - self.fr_measured_ghz) * 1e3, ".-")
            ax_res.axhline(0, color="k", lw=0.5)
            ax_res.set_xlabel("External flux ($\\Phi_0$)")
            ax_res.set_ylabel("Residual (MHz)")
            fig.tight_layout()
            image_path = ds._new_file_path(ds.savefolders[1], self.name, suffix="png")
            fig.savefig(image_path)
            self.figure_paths.append(image_path)

    # --- checks ---------------------------------------------------------------

    def evaluate(self) -> EvaluateResult:
        if self.fit_result is None:
            reason = self.fit_error or "analyze() did not run"
            self.report_output.append(
                f"## Fluxonium Resonator Theory Fit\n**Failed:** {reason}\n"
            )
            logger.warning(f"fit_converged: {reason}")
            return EvaluateResult(
                OperationStatus.FAILURE,
                [CheckResult("fit_converged", False, reason)],
            )

        lm = self.fit_result.lmfit_result
        converged = bool(getattr(lm, "success", False)) and bool(getattr(lm, "errorbars", False))

        max_rms = float(self.max_rms_mhz())
        residual_ok = self.rms_mhz <= max_rms

        g_stderr = self.fit_result.params["g"].stderr
        g_lo, g_hi = self.G_BOUNDS_GHZ
        in_range = g_lo < self.g_fit < g_hi
        precise = g_stderr is not None and self.g_fit != 0 and abs(g_stderr / self.g_fit) < 0.2

        checks = [
            CheckResult("fit_converged", converged,
                        f"lmfit success={getattr(lm, 'success', None)}, "
                        f"errorbars={getattr(lm, 'errorbars', None)}"),
            CheckResult("residual_quality", residual_ok,
                        f"RMS residual {self.rms_mhz:.3f} MHz (max {max_rms:.3f})"),
            CheckResult("params_in_range", in_range and precise,
                        f"g = {self.g_fit * 1e3:.3f} MHz"
                        + (f" +/- {g_stderr * 1e3:.3f}" if g_stderr is not None
                           else " (no stderr)")),
        ]

        msg = (
            "## Fluxonium Resonator Theory Fit\n"
            f"Fixed EC / EL / EJ: {self.EC():.5f} / {self.EL():.5f} / {self.EJ():.5f} GHz\n"
            f"Flux points fitted: {len(self.flux_phi0)}\n"
            f"g  = {self.g_fit * 1e3:.4f} MHz\n"
            f"fr = {self.fr_fit:.6f} GHz\n"
            f"RMS residual: {self.rms_mhz:.4f} MHz\n"
        )
        self.report_output.append(msg)

        status = (OperationStatus.SUCCESS if all(c.passed for c in checks)
                  else OperationStatus.FAILURE)
        for check in checks:
            if not check.passed:
                logger.warning(f"{check.name}: {check.description}")
        return EvaluateResult(status, checks)

    def correct(self, result: EvaluateResult) -> EvaluateResult:
        """Append every figure; the base class attaches only the last one."""
        figures = list(self.figure_paths)
        self.figure_paths.clear()
        for path in figures:
            self.report_output.extend(["\n**Theory fit:**\n", path.resolve()])
        return super().correct(result)
