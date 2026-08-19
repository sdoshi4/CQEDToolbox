"""Locate zero and half flux from a measured resonator-vs-flux sweep.

Analysis-only: it takes no data of its own and instead re-reads what
ResonatorSpectroscopyVsFlux already measured and notch-fitted, runs it through
the symmetry + CNN pipeline in `flux_offset`, and writes
the bias currents the downstream fluxonium operations need.

The pipeline splits the problem in two.  A fold-symmetry search over the
dominant resonance locates the mirror centres of the curve and the flux period;
on real sweeps it is 3-50x more accurate than the network.  But every symmetry
centre is either zero flux or half flux and the two families are geometrically
identical, so no fold can tell them apart -- that single bit comes from the
parity head of a 5-seed CNN ensemble, whose spread is the confidence.  Where a
trace shows a second (e0-e1) notch, its location is an independent check on that
parity call, since the e0-e1 branch is only populated near half flux.  When the
evidence does not line up the pipeline abstains rather than guessing.
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr
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

    # How far outside the window the ensemble was trained on the measured curve
    # may sit, as a fraction of that window's width.  The models learned one
    # device in one 10 MHz band; a curve from outside it does not fail loudly,
    # it returns a confident and wrong parity bit.
    TRAINING_WINDOW_MARGIN = 0.5

    def __init__(self, params, source=None, checkpoint_dir=None,
                 checkpoint_glob="flux_cnn_seed*.pt", freq_to_ghz=1e-3):
        super().__init__()

        # The operation that measured and notch-fitted the flux sweep.  Its
        # analysis output is this operation's input; there is no framework
        # channel for that, so the instance is handed in at construction.
        self.source = source
        # Directory holding the trained ensemble -- machine-local, so it is
        # injected here rather than kept in the parameter manager.  The whole
        # ensemble is needed, not one member: the spread of the seeds' parity
        # probabilities is what the confidence gate reads.
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_glob = checkpoint_glob
        # The network reasons in absolute GHz, so the sweep's native frequency
        # unit has to be converted here and be right.  QICK reports MHz; the
        # dummy platform already emits GHz and needs freq_to_ghz=1.0.
        self.freq_to_ghz = float(freq_to_ghz)

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

        self.condition = ("Success if the symmetry + CNN pipeline reaches a "
                          "conclusion and places zero and half flux inside the "
                          "measured current range")
        # measure() is free here, and a retry re-analyses the same data
        # deterministically, so the default of 100 attempts is meaningless.
        self.max_attempts = 3

        self.result = None
        self.zero_current = None
        self.half_current = None
        self.median_fr_ghz = None
        self.branches = None
        self.sweep = None
        self.models = None
        self.meta = None
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

    # --- inputs from the source operation ------------------------------------

    def _branches_from_source(self):
        """Rebuild the CNN's four channels from the notch fits already made.

        ResonatorSpectroscopyVsFlux.analyze() runs `fit_lorentzian_notches` over
        every trace and keeps the centres and the count, so refitting here would
        redo identical work.  `extract_branches` takes ``centres[0]`` and
        ``centres[-1]``, which for a one-notch trace are the same value -- hence
        f_high falling back to column 0.

        A trace the fit rejected stays in place as ``valid = 0`` rather than
        being dropped: the CNN input is fixed-length and the position of a gap
        carries information.
        """
        cur = np.asarray(self.source.independents["flux"], float)
        counts = np.asarray(self.source.notch_counts, int)
        centres = np.asarray(self.source.notch_centres, float) * self.freq_to_ghz

        valid = counts > 0
        has_second = counts > 1
        f_low = centres[:, 0]
        f_high = np.where(has_second, centres[:, 1], centres[:, 0])
        return cur, f_low, f_high, has_second, valid

    def _sweep_from_source(self, cur):
        """The magnitude map, as the Dataset the symmetry extractor expects.

        Built in memory rather than re-read from the ddh5.  Current stays in its
        native unit: the fold is unit-agnostic in current, and the CNN never sees
        it at all, since channels are built from a period-normalised crop.
        """
        freq = np.asarray(self.source.independents["frequencies"], float)
        mag = np.abs(np.asarray(self.source.dependents["signal"]))
        return xr.Dataset({"signal": xr.DataArray(
            mag,
            coords={"current": cur, "freq": freq * self.freq_to_ghz},
            dims=("current", "freq"),
        )})

    def _load_ensemble(self):
        """Load every checkpoint in `checkpoint_dir`; cached across retries."""
        from flux_offset import load_model

        if self.models:
            return
        if self.checkpoint_dir is None or not self.checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"checkpoint directory not found at {self.checkpoint_dir}"
            )
        paths = sorted(self.checkpoint_dir.glob(self.checkpoint_glob))
        if not paths:
            raise FileNotFoundError(
                f"no checkpoints matching {self.checkpoint_glob!r} in "
                f"{self.checkpoint_dir}"
            )

        models, metas = [], []
        for path in paths:
            model, meta = load_model(path)
            models.append(model)
            metas.append(meta)

        # A mismatched member would silently corrupt the ensemble average.
        for path, meta in zip(paths[1:], metas[1:]):
            for key in ("in_channels", "n_points"):
                if meta.get(key) != metas[0].get(key):
                    raise ValueError(
                        f"{path.name} disagrees with {paths[0].name} on {key} "
                        f"({meta.get(key)} vs {metas[0].get(key)}); the "
                        f"checkpoint directory holds more than one ensemble"
                    )

        self.models, self.meta = models, metas[0]
        logger.info(f"loaded {len(models)} model(s) from {self.checkpoint_dir}")

    # --- analysis -------------------------------------------------------------

    def analyze(self):
        self.result = None
        self.zero_current = self.half_current = None
        self.median_fr_ghz = None
        self.branches = self.sweep = None
        self.load_error = None

        missing = [name for name in ("notch_centres", "notch_counts")
                   if getattr(self.source, name, None) is None]
        if missing:
            self.load_error = (f"source operation produced no notch fits "
                               f"(missing {', '.join(missing)}); it must run "
                               f"analyze() before this operation")
            logger.warning(self.load_error)
            return
        for key, holder in (("flux", self.source.independents),
                            ("frequencies", self.source.independents),
                            ("signal", self.source.dependents)):
            if np.asarray(holder.get(key, [])).size == 0:
                self.load_error = f"source operation has no {key!r} data"
                logger.warning(self.load_error)
                return

        try:
            # Imported here so that this module, and the operations package,
            # stay importable on machines without torch.
            from flux_offset import locate_flux_points
        except ImportError as exc:
            self.load_error = f"flux_offset analysis unavailable ({exc})"
            logger.warning(self.load_error)
            return

        self.branches = self._branches_from_source()
        cur, f_low, _f_high, _has_second, valid = self.branches
        self.sweep = self._sweep_from_source(cur)
        if valid.any():
            median = float(np.nanmedian(f_low[valid]))
            self.median_fr_ghz = median if np.isfinite(median) else None

        try:
            self._load_ensemble()
            self.result = locate_flux_points(
                self.sweep, self.models, branches=self.branches, verbose=False
            )
        except ImportError as exc:
            self.load_error = f"torch is not installed ({exc}); install the 'ml' extra"
            logger.warning(self.load_error)
            return
        except Exception as exc:
            self.load_error = f"inference failed: {exc}"
            logger.warning(self.load_error)
            return

        # Of the equivalent lattice points, take the one needing the least bias
        # current.  The pipeline's own anchor is an arbitrary member of the
        # family and can sit outside the swept range entirely.
        nearest_zero = lambda vals, fallback: (
            min(vals, key=abs) if vals else fallback
        )
        self.zero_current = nearest_zero(self.result.zero_all, self.result.zero_flux)
        self.half_current = nearest_zero(self.result.half_all, self.result.half_flux)
        logger.info(
            f"{self.result.status}: zero={self.zero_current}, "
            f"half={self.half_current}, period={self.result.period}"
        )

        self._save_analysis()

    def _save_analysis(self):
        result = self.result
        entries = dict(
            status=result.status,
            zero_flux_current=self.zero_current,
            half_flux_current=self.half_current,
            zero_all=np.asarray(result.zero_all, float),
            half_all=np.asarray(result.half_all, float),
            period=result.period,
            period_source=result.period_source,
            position_source=result.position_source,
            parity_prob=result.parity_prob,
            parity_spread=result.parity_spread,
            e0e1_center=result.e0e1_center,
            e0e1_agreement=result.e0e1_agreement,
            symmetry_status=(None if result.symmetry is None
                             else result.symmetry.status),
            # a bare list is not a type DatasetAnalysis can save, and it falls
            # back to a pickle nobody will ever read
            notes="\n".join(result.notes),
        )
        # On an abstain most of these are None, which DatasetAnalysis cannot
        # save either -- it would write a pickle per absent field.  Their
        # absence from the saved analysis is the same information.
        entries = {k: v for k, v in entries.items() if v is not None}

        with DatasetAnalysis(self.data_loc.parent, self.name) as ds:
            ds.add(**entries)
            fig = self._make_figure()
            if fig is not None:
                image_path = ds._new_file_path(ds.savefolders[1], self.name,
                                               suffix="png")
                # _new_file_path only builds the name; the folder is not created
                # until DatasetAnalysis.save() runs on __exit__, which is after
                # this write.
                image_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(image_path)
                plt.close(fig)
                self.figure_paths.append(image_path)

    def _make_figure(self):
        """Sweep with the predicted lattices, over the fold residual below it.

        The lower panel is the diagnostic that shows *why* the points sit where
        they do: each detected symmetry centre is a deep minimum of the fold
        mismatch, and a shallow or missing minimum is what a bad result looks
        like.
        """
        result = self.result
        cur, f_low, f_high, has_second, valid = self.branches
        freq = np.asarray(self.sweep.coords["freq"].values, float)
        mag = self.sweep["signal"].transpose("current", "freq").values

        fig, (ax, ax_res) = plt.subplots(
            2, 1, sharex=True, figsize=(11, 7),
            gridspec_kw={"height_ratios": [3, 1]},
        )
        # rasterized: a vector pcolormesh of a few hundred x a hundred cells
        # makes an enormous PNG-embedding report
        ax.pcolormesh(cur, freq, mag.T, shading="auto", cmap="magma",
                      rasterized=True)
        ax.plot(cur[valid], f_low[valid], ".", ms=3, color="tab:cyan",
                label="extracted")
        if has_second.any():
            ax.plot(cur[has_second], f_high[has_second], ".", ms=6,
                    color="tab:red", label="second notch (e0-e1)")
        for j, v in enumerate(result.zero_all):
            ax.axvline(v, color="tab:green", lw=2,
                       label="zero flux" if j == 0 else None)
        for j, v in enumerate(result.half_all):
            ax.axvline(v, color="tab:orange", lw=2,
                       label="half flux" if j == 0 else None)
        ax.set_ylabel("Readout frequency (GHz)")
        ax.set_title(
            f"{result.status.upper()} - {result.position_source or 'no positions'}"
            + (f", parity p={result.parity_prob:.3f}"
               if result.parity_prob is not None else "")
        )
        ax.legend(fontsize=8, loc="upper right")

        sym = result.symmetry_result
        if sym is not None:
            ax_res.semilogy(sym.grid, sym.residual, "-", lw=1, color="0.3")
            ax_res.plot(sym.centers, sym.center_res, "v", ms=7,
                        color="tab:blue", label="detected centres")
            ax_res.axhline(0.10, color="k", ls=":", lw=0.8,
                           label="acceptance threshold")
            ax_res.legend(fontsize=7, loc="upper right")
            ax_res.set_ylabel("fold residual R")
        for v in result.zero_all:
            ax_res.axvline(v, color="tab:green", lw=1.5)
        for v in result.half_all:
            ax_res.axvline(v, color="tab:orange", lw=1.5)
        ax_res.set_xlabel("Flux current (uA)")
        fig.tight_layout()
        return fig

    # --- checks ---------------------------------------------------------------

    def evaluate(self) -> EvaluateResult:
        from flux_offset import STATUS_INCONCLUSIVE

        if self.result is None:
            reason = self.load_error or "analyze() did not run"
            self.report_output.append(f"## Flux Offset Inference\n**Failed:** {reason}\n")
            logger.warning(f"pipeline_conclusive: {reason}")
            return EvaluateResult(
                OperationStatus.FAILURE,
                [CheckResult("pipeline_conclusive", False, reason)],
            )

        result = self.result
        conclusive = result.status != STATUS_INCONCLUSIVE

        # The band the ensemble was actually trained on, not the parameter
        # manager's bare_fr: a curve outside it gets a confident, wrong answer.
        acq = (self.meta or {}).get("acq", {})
        f_center, f_span = acq.get("f_center"), acq.get("f_span")
        if f_center is None or f_span is None or self.median_fr_ghz is None:
            in_window = False
            window_msg = ("checkpoint carries no training window, or no trace "
                          "was fitted; cannot verify the model is in range")
        else:
            half = 0.5 * f_span * (1.0 + 2.0 * self.TRAINING_WINDOW_MARGIN)
            in_window = abs(self.median_fr_ghz - f_center) <= half
            window_msg = (
                f"curve median {self.median_fr_ghz:.5f} GHz vs training window "
                f"{f_center:.5f} +/- {1e3 * half:.1f} MHz "
                f"(bare_fr parameter says {float(self.bare_fr()):.5f} GHz)"
            )

        lo, hi = sorted((float(self.start_flux()), float(self.end_flux())))
        in_range = (self.zero_current is not None
                    and self.half_current is not None
                    and lo <= self.zero_current <= hi
                    and lo <= self.half_current <= hi)

        checks = [
            CheckResult("pipeline_conclusive", conclusive,
                        f"pipeline returned {result.status.upper()}"
                        + ("" if conclusive else
                           "; " + "; ".join(result.notes))),
            CheckResult("training_window", in_window, window_msg),
            CheckResult("markers_in_range", in_range,
                        f"zero={self.zero_current}, half={self.half_current}, "
                        f"swept range [{lo:.6g}, {hi:.6g}]"),
        ]

        self.report_output.append(self._report_block(result))

        status = (OperationStatus.SUCCESS if all(c.passed for c in checks)
                  else OperationStatus.FAILURE)
        for check in checks:
            if not check.passed:
                logger.warning(f"{check.name}: {check.description}")
        return EvaluateResult(status, checks)

    def _report_block(self, result) -> str:
        from flux_offset import STATUS_OK

        n_models = len(self.models) if self.models else 0
        lines = [
            "## Flux Offset Inference",
            f"Ensemble: {n_models} model(s) from `{self.checkpoint_dir}`",
            f"Result: **{result.status.upper()}**",
        ]
        if result.status != STATUS_OK:
            lines.append(
                "> An `uncertain` result still writes the flux points, but at "
                "least one piece of evidence did not line up; the notes below "
                "say which."
            )
        if self.zero_current is not None:
            lines += [
                f"Zero flux current: {self.zero_current:.6g}",
                f"Half flux current: {self.half_current:.6g}",
                f"All zero flux points: "
                f"{[round(v, 6) for v in result.zero_all]}",
                f"All half flux points: "
                f"{[round(v, 6) for v in result.half_all]}",
            ]
        if result.period is not None:
            lines.append(f"Period: {result.period:.6g} [{result.period_source}]")
        if result.position_source:
            lines.append(f"Positions from: {result.position_source}")
        if result.parity_prob is not None:
            lines.append(
                f"Parity: p={result.parity_prob:.3f} "
                f"(ensemble spread {result.parity_spread:.3f})"
                + (f", e0-e1 region agrees to {result.e0e1_agreement:.3f} periods"
                   if result.e0e1_agreement is not None
                   else ", no usable e0-e1 evidence")
            )
        if self.median_fr_ghz is not None:
            lines.append(f"Curve median: {self.median_fr_ghz:.5f} GHz")
        if result.notes:
            lines.append("\nNotes:")
            lines += [f"- {note}" for note in result.notes]
        return "\n".join(lines) + "\n"

    def correct(self, result: EvaluateResult) -> EvaluateResult:
        """Append every figure; the base class attaches only the last one."""
        figures = list(self.figure_paths)
        self.figure_paths.clear()
        for path in figures:
            self.report_output.extend(["\n**Predicted flux markers:**\n", path.resolve()])
        return super().correct(result)
