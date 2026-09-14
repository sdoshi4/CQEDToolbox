"""Fit EJ/EC/EL to the measured qubit line from saturation spectroscopy vs flux.

Analysis-only: it reuses the f01(current) points the sweeps already extracted,
fits the fluxonium spectrum to them, and writes the result back so the next
sweep aims its pump window with corrected parameters.

Run it twice.  The first pass sees only the half-flux sweep and exists to
rescue the zero-flux prediction, which is otherwise off by hundreds of MHz;
the second sees both curves and produces the final numbers.
"""

import logging
from pathlib import Path

import lmfit
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis
from labcore.protocols.base import (ProtocolOperation, OperationStatus,
                                    CheckResult, EvaluateResult)
from parameters import (
    ECParam, ELParam, EJParam, ZeroFluxCurrent, HalfFluxCurrent,
)
from .fluxonium_spectrum import FluxoniumQubitFit
from .fluxonium_theory_fit import TheoryFitMaxRms

logger = logging.getLogger(__name__)

#: Smallest flux span, in flux quanta, that identifies EJ/EC/EL well enough to
#: extrapolate.  Not a tuning knob -- below it the fit is degenerate, and the
#: residual does not reveal it: on synthetic data spanning 0.042 quanta the fit
#: returned EJ=6.7/EC=2.0 against a truth of 4.0/1.0 with a 0.35 MHz residual,
#: mispredicting f01 at zero flux by 3.3 GHz.  At 0.2 quanta the same fit is
#: accurate to 7 MHz.  This check, not the residual, is what catches that.
MIN_DELTA_PHI = 0.15


class FluxoniumQubitTheoryFit(ProtocolOperation):

    # Each model evaluation diagonalizes one fluxonium per flux point and lmfit
    # needs hundreds of them, so the pooled curve is downsampled.
    N_THEORY = 41
    # How far the fitted half-flux current may move from the value
    # FluxOffsetInference wrote, uA.  Its own quoted accuracy is 0.3-5.1 uA.
    I_HALF_SEARCH_UA = 15.0

    def __init__(self, params, sources=None, label=""):
        super().__init__()

        # The sweeps whose extracted qubit line this fits; their analyze() has
        # already produced `currents`, `peak_freq` and `found`.
        self.sources = list(sources or [])
        # Both passes are the same class, so without a label their analysis
        # folders and report sections would collide.
        if label:
            self.name = f"{self.name}_{label}"

        self._register_inputs(
            zero_flux_current=ZeroFluxCurrent(params),
        )
        self._register_outputs(
            EC=ECParam(params),
            EL=ELParam(params),
            EJ=EJParam(params),
            half_flux_current=HalfFluxCurrent(params),
        )
        self._register_correction_params(
            max_rms_mhz=TheoryFitMaxRms(params),
        )

        self._register_success_update(self.EJ, lambda: float(self.EJ_fit))
        self._register_success_update(self.EC, lambda: float(self.EC_fit))
        self._register_success_update(self.EL, lambda: float(self.EL_fit))
        self._register_success_update(self.half_flux_current,
                                      lambda: float(self.I_half_fit))

        self.condition = ("Success if the fluxonium spectrum fits the measured "
                          "qubit line over a wide enough flux range to be "
                          "identifiable")
        self.max_attempts = 3

        self.fit_result = None
        self.currents = None
        self.f01_measured = None
        self.f01_model = None
        self.flux_phi0 = None
        self.EJ_fit = self.EC_fit = self.EL_fit = self.I_half_fit = None
        self.rms_mhz = None
        self.delta_phi = None
        self.fit_error = None

    # --- no measurement of its own -------------------------------------------

    def _measure_qick(self) -> Path:
        return self._reuse_source_data()

    def _measure_dummy(self) -> Path:
        return self._reuse_source_data()

    def _reuse_source_data(self) -> Path:
        if not self.sources or self.sources[-1].data_loc is None:
            raise RuntimeError(
                "FluxoniumQubitTheoryFit needs the SaturationSpectroscopyVsFlux "
                "operations it follows; pass sources=[<those operations>]."
            )
        return self.sources[-1].data_loc

    def _load_data_qick(self):
        pass

    def _load_data_dummy(self):
        pass

    # --- analysis -------------------------------------------------------------

    def _pooled_curve(self):
        """Every fitted (current, f01) point from every source, in GHz."""
        currents, freqs = [], []
        for source in self.sources:
            found = getattr(source, "found", None)
            if found is None or not np.any(found):
                continue
            currents.append(np.asarray(source.currents, float)[found])
            freqs.append(np.asarray(source.peak_freq, float)[found] * 1e-3)
        if not currents:
            return None, None
        currents = np.concatenate(currents)
        freqs = np.concatenate(freqs)
        order = np.argsort(currents)
        return currents[order], freqs[order]

    def analyze(self):
        self.fit_result = None
        self.EJ_fit = self.EC_fit = self.EL_fit = self.I_half_fit = None
        self.rms_mhz = self.delta_phi = None
        self.fit_error = None

        currents, f01 = self._pooled_curve()
        if currents is None or len(currents) < 6:
            self.fit_error = "sources produced fewer than 6 fitted qubit points"
            logger.warning(self.fit_error)
            return

        zero = float(self.zero_flux_current())
        half = float(self.half_flux_current())
        period = 2 * (half - zero)
        if not np.isfinite(period) or period == 0:
            self.fit_error = (f"flux calibration is unusable (zero={zero}, "
                              f"half={half}); the flux axis cannot be built")
            logger.warning(self.fit_error)
            return

        # Evenly spaced subset, so the fit sees the whole flux range rather than
        # whichever window contributed the most points.
        if len(currents) > self.N_THEORY:
            keep = np.unique(np.linspace(0, len(currents) - 1, self.N_THEORY).astype(int))
            currents, f01 = currents[keep], f01[keep]
        self.currents, self.f01_measured = currents, f01
        self.delta_phi = float(np.ptp(currents) / abs(period))

        fit = FluxoniumQubitFit(currents, f01)
        fit.period = period
        # I_half floats because FluxOffsetInference only locates it to a few uA,
        # and holding it fixed at even its best case leaves a 13 MHz residual
        # against a 1 MHz threshold; the curve's symmetry pins it far better.
        fit_params = {
            "EJ": lmfit.Parameter("EJ", value=float(self.EJ()), min=0.5, max=20.0),
            "ECq": lmfit.Parameter("ECq", value=float(self.EC()), min=0.1, max=5.0),
            "ELq": lmfit.Parameter("ELq", value=float(self.EL()), min=0.05, max=5.0),
            "I_half": lmfit.Parameter("I_half", value=half,
                                      min=half - self.I_HALF_SEARCH_UA,
                                      max=half + self.I_HALF_SEARCH_UA),
        }

        logger.info(f"Fitting {len(currents)} qubit points spanning "
                    f"{self.delta_phi:.3f} flux quanta")
        try:
            self.fit_result = fit.run(params=fit_params)
        except Exception as exc:
            self.fit_error = f"qubit theory fit failed: {exc}"
            logger.warning(self.fit_error)
            return

        p = self.fit_result.params
        self.EJ_fit = float(p["EJ"].value)
        self.EC_fit = float(p["ECq"].value)
        self.EL_fit = float(p["ELq"].value)
        self.I_half_fit = float(p["I_half"].value)
        self.f01_model = np.asarray(self.fit_result.eval(), dtype=float)
        self.rms_mhz = float(np.sqrt(np.mean((self.f01_model - f01) ** 2)) * 1e3)
        self.flux_phi0 = 0.5 + (currents - self.I_half_fit) / period
        logger.info(f"EJ={self.EJ_fit:.4f}, EC={self.EC_fit:.4f}, "
                    f"EL={self.EL_fit:.4f} GHz, I_half={self.I_half_fit:.3f} uA "
                    f"({self.I_half_fit - half:+.3f} uA), RMS={self.rms_mhz:.3f} MHz")

        self._save_analysis()

    def _save_analysis(self):
        with DatasetAnalysis(self.data_loc.parent, self.name) as ds:
            ds.add(
                current_uA=self.currents,
                flux_phi0=self.flux_phi0,
                f01_measured_ghz=self.f01_measured,
                f01_model_ghz=self.f01_model,
                EJ_ghz=self.EJ_fit, EC_ghz=self.EC_fit, EL_ghz=self.EL_fit,
                half_flux_current_uA=self.I_half_fit,
                delta_phi=self.delta_phi,
                rms_residual_mhz=self.rms_mhz,
            )
            fig, (ax, ax_res) = plt.subplots(
                2, 1, sharex=True, figsize=(7, 6),
                gridspec_kw={"height_ratios": [3, 1]},
            )
            ax.plot(self.flux_phi0, self.f01_measured, "o", ms=4, label="measured")
            ax.plot(self.flux_phi0, self.f01_model, "-", label="theory fit")
            ax.set_ylabel("Qubit frequency (GHz)")
            ax.set_title(f"EJ={self.EJ_fit:.4f}, EC={self.EC_fit:.4f}, "
                         f"EL={self.EL_fit:.4f} GHz")
            ax.legend(fontsize="small")
            ax_res.plot(self.flux_phi0,
                        (self.f01_model - self.f01_measured) * 1e3, ".-")
            ax_res.axhline(0, color="k", lw=0.5)
            ax_res.set_xlabel("External flux ($\\Phi_0$)")
            ax_res.set_ylabel("Residual (MHz)")
            fig.tight_layout()
            image_path = ds._new_file_path(ds.savefolders[1], self.name, suffix="png")
            fig.savefig(image_path)
            plt.close(fig)
            self.figure_paths.append(image_path)

    # --- checks ---------------------------------------------------------------

    def evaluate(self) -> EvaluateResult:
        if self.fit_result is None:
            reason = self.fit_error or "analyze() did not run"
            self.report_output.append(
                f"## Fluxonium Qubit Theory Fit\n**Failed:** {reason}\n"
            )
            logger.warning(f"fit_converged: {reason}")
            return EvaluateResult(
                OperationStatus.FAILURE,
                [CheckResult("fit_converged", False, reason)],
            )

        lm = self.fit_result.lmfit_result
        checks = [
            CheckResult(
                "fit_converged",
                bool(getattr(lm, "success", False)) and bool(getattr(lm, "errorbars", False)),
                f"lmfit success={getattr(lm, 'success', None)}, "
                f"errorbars={getattr(lm, 'errorbars', None)}",
            ),
            # A wide enough flux range is the only thing that makes EJ/EC/EL
            # identifiable; a small residual on a narrow range does not.
            CheckResult(
                "flux_coverage", self.delta_phi >= MIN_DELTA_PHI,
                f"fitted points span {self.delta_phi:.3f} flux quanta "
                f"(need {MIN_DELTA_PHI}); below this the fit is degenerate and "
                f"the residual will not show it",
            ),
            CheckResult(
                "fit_residual", self.rms_mhz <= float(self.max_rms_mhz()),
                f"RMS residual {self.rms_mhz:.3f} MHz "
                f"(limit {float(self.max_rms_mhz()):.3f})",
            ),
        ]

        self.report_output.append(self._report_block())
        for check in checks:
            if not check.passed:
                logger.warning(f"{check.name}: {check.description}")
        return EvaluateResult(
            OperationStatus.SUCCESS if all(c.passed for c in checks)
            else OperationStatus.FAILURE,
            checks,
        )

    def _report_block(self) -> str:
        moved = self.I_half_fit - float(self.half_flux_current())
        return "\n".join([
            "## Fluxonium Qubit Theory Fit",
            f"Fitted {len(self.currents)} points from "
            f"{len(self.sources)} sweep(s), spanning {self.delta_phi:.3f} $\\Phi_0$",
            f"EJ = {self.EJ_fit:.5f} GHz",
            f"EC = {self.EC_fit:.5f} GHz",
            f"EL = {self.EL_fit:.5f} GHz",
            f"Half flux current: {self.I_half_fit:.3f} uA ({moved:+.3f} uA)",
            f"RMS residual: {self.rms_mhz:.3f} MHz",
        ]) + "\n"

    def correct(self, result: EvaluateResult) -> EvaluateResult:
        """Attach the fit figure before the base class adds the check table."""
        figures = list(self.figure_paths)
        self.figure_paths.clear()
        for path in figures:
            self.report_output.extend(["\n**Qubit spectrum fit:**\n", path.resolve()])
        return super().correct(result)
