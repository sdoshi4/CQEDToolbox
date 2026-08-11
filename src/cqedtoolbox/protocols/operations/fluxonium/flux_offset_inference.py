"""Recover the flux offset from a measured resonator-vs-flux curve.

Analysis-only: it takes no data of its own, and instead consumes the one-period
curve that ResonatorSpectroscopyVsFlux already extracted, runs it through the
trained offset-inverse CNN, and converts the predicted offset into the zero- and
half-flux bias currents that the downstream fluxonium operations need.
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from labcore.analysis import DatasetAnalysis
from labcore.protocols.base import (ProtocolOperation, OperationStatus,
                                    CheckResult, EvaluateResult)
from parameters import (
    StartFlux, EndFlux, ZeroFluxCurrent, HalfFluxCurrent, ResonatorFr,
)

logger = logging.getLogger(__name__)


class FluxOffsetInference(ProtocolOperation):

    # Fraction of the bare resonator frequency the measured curve is allowed to
    # sit away from it before we assume something is wrong with the units or the
    # model's training range.  The CNN was trained on a narrow band around one
    # device; feeding it a curve from far outside that band returns a confident
    # and wrong answer.
    FREQUENCY_BAND_TOLERANCE = 0.05

    def __init__(self, params, source=None, checkpoint_path=None):
        super().__init__()

        # The operation that measured and preprocessed the flux sweep.  Its
        # analysis output is this operation's input; there is no framework
        # channel for that, so the instance is handed in at construction.
        self.source = source
        # Filesystem location of the trained .pt -- machine-local, so it is
        # injected here rather than kept in the parameter manager.
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        self._register_inputs(
            start_flux=StartFlux(params),
            end_flux=EndFlux(params),
            bare_fr=ResonatorFr(params),
        )
        self._register_outputs(
            zero_flux_current=ZeroFluxCurrent(params),
            half_flux_current=HalfFluxCurrent(params),
        )

        self._register_success_update(
            self.zero_flux_current, lambda: float(self.zero_current)
        )
        self._register_success_update(
            self.half_flux_current, lambda: float(self.half_current)
        )

        self.condition = ("Success if the CNN returns an offset that places zero "
                          "and half flux inside the measured current range")
        # measure() is free here, so the default of 100 attempts is far too many.
        self.max_attempts = 3

        self.prediction = None
        self.offset_fraction = None
        self.zero_current = None
        self.half_current = None
        self.curve_ghz = None
        self.load_error = None

    # --- no measurement of its own -------------------------------------------
    # Reusing the source's data_loc puts this operation's figures and analysis
    # into the same dataset folder, under its own name.

    def _measure_qick(self) -> Path:
        return self._reuse_source_data()

    def _measure_dummy(self) -> Path:
        return self._reuse_source_data()

    def _reuse_source_data(self) -> Path:
        if self.source is None or self.source.data_loc is None:
            raise RuntimeError(
                "FluxOffsetInference needs the ResonatorSpectroscopyVsFlux "
                "operation it follows; pass source=<that operation>."
            )
        return self.source.data_loc

    def _load_data_qick(self):
        pass

    def _load_data_dummy(self):
        pass

    # --- analysis -------------------------------------------------------------

    def analyze(self):
        self.prediction = None
        self.offset_fraction = None
        self.zero_current = self.half_current = None
        self.curve_ghz = None
        self.load_error = None

        curve = getattr(self.source, "fr_resampled", None)
        flux_resampled = getattr(self.source, "flux_resampled", None)
        crop_span = getattr(self.source, "crop_span", None)
        if curve is None or flux_resampled is None or crop_span is None:
            self.load_error = "source operation produced no one-period curve"
            logger.warning(self.load_error)
            return

        # The network sees absolute frequency, so the conversion out of the
        # measuring operation's native units has to happen here and be right.
        self.curve_ghz = np.asarray(curve, dtype=float) * self.source.FREQ_UNIT_TO_GHZ

        try:
            # Imported here so that this module, and the operations package,
            # stay importable on machines without torch.
            from cqedtoolbox.analysis.fluxonium_inverse_cnn import (
                load_checkpoint, predict_from_checkpoint, offset_fraction_of,
            )
        except ImportError as exc:
            self.load_error = f"torch is not installed ({exc}); install the 'ml' extra"
            logger.warning(self.load_error)
            return

        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            self.load_error = f"checkpoint not found at {self.checkpoint_path}"
            logger.warning(self.load_error)
            return

        try:
            ckpt = load_checkpoint(self.checkpoint_path)
            self.prediction = predict_from_checkpoint(self.curve_ghz, ckpt)
            self.offset_fraction = offset_fraction_of(self.prediction)
        except Exception as exc:
            self.load_error = f"inference failed: {exc}"
            logger.warning(self.load_error)
            return

        # The resampled grid spans crop_span, starting at flux_resampled[0].
        # Using the crop window's own width instead would overstate the span by
        # up to one sample gap and bias both markers.
        origin = float(flux_resampled[0])
        self.zero_current = origin + (self.offset_fraction % 1.0) * crop_span
        self.half_current = origin + ((self.offset_fraction + 0.5) % 1.0) * crop_span
        logger.info(
            f"offset={self.offset_fraction:.4f} -> zero={self.zero_current:.6g}, "
            f"half={self.half_current:.6g}"
        )

        with DatasetAnalysis(self.data_loc.parent, self.name) as ds:
            ds.add(
                offset_fraction=float(self.offset_fraction),
                zero_flux_current=float(self.zero_current),
                half_flux_current=float(self.half_current),
                curve_ghz=self.curve_ghz,
                prediction=dict(self.prediction),
            )

            fig, ax = plt.subplots()
            ax.plot(flux_resampled, self.curve_ghz, ".-", ms=3,
                    label=f"one period ({len(self.curve_ghz)} pts)")
            ax.axvline(self.zero_current, color="red", lw=2, label="zero flux")
            ax.axvline(self.half_current, color="orange", lw=2, label="half flux")
            ax.set_xlabel("Flux / Current")
            ax.set_ylabel("Resonator frequency (GHz)")
            ax.set_title(f"Predicted flux offset = {self.offset_fraction:.4f}")
            ax.legend(fontsize="small")
            image_path = ds._new_file_path(ds.savefolders[1], self.name, suffix="png")
            fig.savefig(image_path)
            self.figure_paths.append(image_path)

    # --- checks ---------------------------------------------------------------

    def evaluate(self) -> EvaluateResult:
        if self.offset_fraction is None:
            reason = self.load_error or "analyze() did not run"
            self.report_output.append(f"## Flux Offset Inference\n**Failed:** {reason}\n")
            logger.warning(f"curve_available: {reason}")
            return EvaluateResult(
                OperationStatus.FAILURE,
                [CheckResult("curve_available", False, reason)],
            )

        expected = float(self.bare_fr())
        median_ghz = float(np.median(self.curve_ghz))
        band = self.FREQUENCY_BAND_TOLERANCE * expected
        in_band = abs(median_ghz - expected) <= band

        lo, hi = sorted((float(self.start_flux()), float(self.end_flux())))
        in_range = (lo <= self.zero_current <= hi) and (lo <= self.half_current <= hi)

        checks = [
            CheckResult("curve_available", True,
                        f"{len(self.curve_ghz)}-point curve, inference returned "
                        f"offset={self.offset_fraction:.4f}"),
            CheckResult("frequency_band", in_band,
                        f"curve median {median_ghz:.4f} GHz vs bare_fr {expected:.4f} GHz "
                        f"(tolerance +/-{band:.4f})"),
            CheckResult("markers_in_range", in_range,
                        f"zero={self.zero_current:.6g}, half={self.half_current:.6g}, "
                        f"swept range [{lo:.6g}, {hi:.6g}]"),
        ]

        msg = (
            "## Flux Offset Inference\n"
            f"Checkpoint: `{self.checkpoint_path}`\n"
            f"Offset fraction: {self.offset_fraction:.6f}\n"
            f"Zero flux current: {self.zero_current:.6g}\n"
            f"Half flux current: {self.half_current:.6g}\n"
            f"Curve median: {median_ghz:.4f} GHz\n"
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
            self.report_output.extend(["\n**Predicted flux markers:**\n", path.resolve()])
        return super().correct(result)
