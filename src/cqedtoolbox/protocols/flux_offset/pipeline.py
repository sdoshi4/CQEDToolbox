"""One call that turns a resonator-vs-flux sweep into zero/half flux points.

    sweep -> extract branches -> symmetry (positions + period)
                             -> CNN (parity, and positions only as a fallback)
                             -> flux points, or an explicit abstain

Division of labour, from measurement on four real sweeps:

  * symmetry locates the points 3-50x better than the CNN (0.3-5.1 uA against
    13.7-15.8 uA, judged against the independent e0-e1 region centre), and the
    CNN carries a systematic ~-0.023 period bias.  So symmetry owns *positions*.
  * symmetry cannot say which family is zero flux and which is half -- no
    geometric test can, since both families are internally self-identical.  The
    CNN's parity head can, and did 4/4.  So the CNN owns *parity*.

Vendored from the `Fluxonium-offset-inverse-model` repository
(`flux_pipeline.py`).  The only functional change is the `branches` argument to
`locate_flux_points`, which lets a caller that has already run
`fit_lorentzian_notches` hand the result in rather than paying for it twice.
"""

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

from . import cnn as F
from .symmetry import (
    find_symmetry, filter_outliers, symmetry_quality,
    extract_resonator_signal, dominant_series,
    SymmetryQuality, SymmetryResult, SYM_OK,
)

STATUS_OK = "ok"
STATUS_UNCERTAIN = "uncertain"
STATUS_INCONCLUSIVE = "inconclusive"

# The CNN's measured systematic bias on real sweeps, in periods.  Recorded, not
# corrected: three sweeps across two qubits is not enough to calibrate a
# correction, and a wrong correction is worse than a documented known one.
CNN_KNOWN_BIAS_PERIODS = -0.023


@dataclass
class FluxPoints:
    """Everything the pipeline concluded, including how it got there."""
    status: str
    zero_flux: Optional[float] = None          # a representative zero-flux current
    half_flux: Optional[float] = None
    zero_all: List[float] = field(default_factory=list)   # all equivalents in range
    half_all: List[float] = field(default_factory=list)
    period: Optional[float] = None
    period_source: str = ""                    # "symmetry lattice" | "translational"
    position_source: str = ""                  # "symmetry centres" | "CNN head A"
    parity_prob: Optional[float] = None
    parity_spread: Optional[float] = None
    e0e1_center: Optional[float] = None
    e0e1_agreement: Optional[float] = None     # |CNN half - e0e1 centre| in periods
    symmetry: Optional[SymmetryQuality] = None
    # The full search behind `symmetry`, kept so a caller can plot the fold
    # residual -- the one picture that shows *why* the centres are where they
    # are, and what a bad result looks like (shallow or absent minima).
    symmetry_result: Optional[SymmetryResult] = None
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.status == STATUS_INCONCLUSIVE:
            return (f"INCONCLUSIVE\n  " + "\n  ".join(self.notes))
        return (
            f"{self.status.upper()}\n"
            f"  zero flux @ {self.zero_flux:.6g}   half flux @ {self.half_flux:.6g}\n"
            f"  all zero  : {[round(v, 6) for v in self.zero_all]}\n"
            f"  all half  : {[round(v, 6) for v in self.half_all]}\n"
            f"  period    : {self.period:.6g}  [{self.period_source}]\n"
            f"  positions : {self.position_source}\n"
            f"  parity    : p={self.parity_prob:.3f} (spread {self.parity_spread:.3f})"
            + (f", e0-e1 agrees to {self.e0e1_agreement:.3f} P"
               if self.e0e1_agreement is not None else ", no e0-e1 evidence")
            + ("\n  notes     : " + "\n              ".join(self.notes)
               if self.notes else "")
        )


# ----------------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------------

def extract_branches(ds, fit_lorentzian_notches):
    """Per-current f_low / f_high / has_second / valid.

    `fit_lorentzian_notches` is passed in rather than imported so that whichever
    copy the caller already trusts is the one that runs.

    A trace the extractor rejects becomes `valid = 0` rather than being dropped:
    on a fixed-length CNN input the gap has to stay where it is, and its position
    carries information.
    """
    cur = np.asarray(ds.coords["current"].values, float)
    frq = np.asarray(ds.coords["freq"].values, float)
    mag = ds["signal"].transpose("current", "freq").values

    n = cur.size
    f_low = np.full(n, np.nan)
    f_high = np.full(n, np.nan)
    has_second = np.zeros(n, bool)
    valid = np.zeros(n, bool)

    for i in range(n):
        try:
            r = fit_lorentzian_notches(frq, mag[i])
        except (ValueError, RuntimeError, FloatingPointError):
            continue
        c = np.asarray(r["centres"], float)
        f_low[i], f_high[i] = c[0], c[-1]
        has_second[i] = r["n_notches"] > 1
        valid[i] = True

    return cur, f_low, f_high, has_second, valid


def _symmetry_series(ds):
    """The single curve the symmetry search folds: the *dominant* resonance.

    Deliberately not `f_low` from `fit_lorentzian_notches`.  Near half flux the
    lowest-frequency notch is the e0-e1 line, so `f_low` switches branch there and
    the fold breaks: folding it finds 0 centres on aug11-140900, where folding the
    dominant resonance finds 2.  The dominant (largest-amplitude) branch is the
    g0-g1 line throughout, since p_g > p_e always, so it stays continuous.

    This is why the two extractors are both run: `extract_resonator_signal` reports
    amplitudes and so can pick the dominant branch, while `fit_lorentzian_notches`
    supplies the sorted two-branch channels the CNN consumes.
    """
    res = extract_resonator_signal(ds)
    x, y = dominant_series(res)
    keep, _ = filter_outliers(x, y, np.ones(x.size, bool))
    return x, y, keep


# ----------------------------------------------------------------------------
# Assembling the answer
# ----------------------------------------------------------------------------

def _lattice_in_range(anchor, spacing, lo, hi):
    """All points anchor + n*spacing that fall inside [lo, hi]."""
    if spacing <= 0:
        return []
    n_lo = int(np.floor((lo - anchor) / spacing)) - 1
    n_hi = int(np.ceil((hi - anchor) / spacing)) + 1
    return [anchor + n * spacing for n in range(n_lo, n_hi + 1)
            if lo <= anchor + n * spacing <= hi]


def _transfer_parity(cnn_symmetry_current, centers, period):
    """Map the CNN's parity bit onto the symmetry centre *lattice*.

    Head B's bit refers to head A's own symmetry point, not to any centre the fold
    search happened to detect, so it must be carried across.  Snap head A to the
    lattice `c0 + n*(P/2)` rather than to the nearest *detected* centre: the search
    only reports centres that are foldable, so a sweep can easily have no detected
    centre near head A even though the lattice point exists.  Matching against
    detected centres reported a 0.740 P miss on the June sweep, which is impossible
    for a lattice spaced P/2 -- it just meant the nearest centre was two sites away.

    Returns (lattice_point, match_distance_in_periods); the match is <= 0.25 by
    construction, and against the CNN's ~0.023 P bias it should be far smaller.
    """
    c0 = float(np.asarray(centers, float)[0])
    half = period / 2.0
    n = int(round((cnn_symmetry_current - c0) / half))
    lattice_pt = c0 + n * half
    return lattice_pt, float(abs(cnn_symmetry_current - lattice_pt) / period)


def _finish(out: "FluxPoints", verbose: bool) -> "FluxPoints":
    if verbose:
        print(out.summary())
    return out


def locate_flux_points(ds, models, fit_lorentzian_notches=None, *,
                       branches=None,
                       parity_conf_min: float = 0.75,
                       e0e1_disagree_max: float = 0.15,
                       verbose: bool = True) -> FluxPoints:
    """Full pipeline: sweep in, flux points (or an explicit abstain) out.

    Pass `branches` as an already-extracted
    ``(cur, f_low, f_high, has_second, valid)`` tuple to skip the per-trace notch
    fit -- a caller that ran `fit_lorentzian_notches` over the same sweep has
    already paid for it.  Otherwise supply `fit_lorentzian_notches` and the fit
    runs here.
    """
    notes: List[str] = []
    degrade = False
    if branches is None:
        if fit_lorentzian_notches is None:
            raise ValueError("supply either `branches` or `fit_lorentzian_notches`")
        branches = extract_branches(ds, fit_lorentzian_notches)
    cur, f_low, f_high, has_second, valid = branches
    cur = np.asarray(cur, float)
    valid = np.asarray(valid, bool)
    has_second = np.asarray(has_second, bool)
    if valid.sum() < 20:
        notes.append(f"only {int(valid.sum())} traces fitted; nothing to work with")
        return _finish(FluxPoints(STATUS_INCONCLUSIVE, notes=notes), verbose)

    x, y, keep = _symmetry_series(ds)
    sym = find_symmetry(x, y, keep, verbose=False)
    q = symmetry_quality(sym)

    # -- period ---------------------------------------------------------------
    if q.usable:
        period, period_source = sym.period, "symmetry lattice"
    elif sym.boot_period:
        period, period_source = sym.boot_period, "translational (symmetry failed)"
        degrade = True
        notes.append("symmetry failed; period from the masked translational scan")
    else:
        notes += q.reasons + ["no period from either method"]
        return _finish(FluxPoints(STATUS_INCONCLUSIVE, symmetry=q,
                                  symmetry_result=sym, notes=notes), verbose)

    # -- CNN ------------------------------------------------------------------
    x_start = float(cur[valid].min())
    channels = F.channels_from_series(cur, f_low, f_high, has_second, valid,
                                      x_start, period)
    pred = F.predict(models, channels)
    cnn_zero = x_start + pred["zero_offset"] * period
    cnn_half = x_start + pred["half_offset"] * period
    cnn_sym = x_start + pred["symmetry_offset"] * period

    lo, hi = float(cur.min()), float(cur.max())

    # -- positions ------------------------------------------------------------
    if q.usable:
        position_source = "symmetry centres"
        anchor, match = _transfer_parity(cnn_sym, sym.centers, period)
        if match > 0.125:        # half of the P/4 that would make it ambiguous
            degrade = True
            notes.append(f"CNN symmetry point is {match:.3f} P from the lattice; "
                         f"parity transfer is not clean")
        # head B says whether head A's point -- hence this lattice site -- is zero
        if pred["parity_prob"] >= 0.5:
            zero_anchor, half_anchor = anchor, anchor + period / 2
        else:
            zero_anchor, half_anchor = anchor + period / 2, anchor
    else:
        position_source = "CNN head A (symmetry failed)"
        notes.append(f"positions carry the CNN's known ~{CNN_KNOWN_BIAS_PERIODS:+.3f} "
                     f"period bias")
        zero_anchor, half_anchor = cnn_zero, cnn_half

    zero_all = _lattice_in_range(zero_anchor, period, lo, hi)
    half_all = _lattice_in_range(half_anchor, period, lo, hi)

    # -- independent parity cross-check --------------------------------------
    e0e1_center = e0e1_agree = None
    n_two = int(has_second.sum())
    if n_two:
        span = (cur[has_second].max() - cur[has_second].min()) / period
    if n_two and span > 0.40:
        notes.append(f"e0-e1 traces are spread over {span:.2f} P, too scattered to "
                     f"locate half flux; cross-check skipped")
    elif n_two:
        e0e1_center = float(np.median(cur[has_second]))
        d_half = min((abs(e0e1_center - v) for v in half_all), default=np.inf)
        d_zero = min((abs(e0e1_center - v) for v in zero_all), default=np.inf)
        e0e1_agree = float(d_half / period)
        if d_zero < d_half:
            degrade = True
            notes.append(
                f"e0-e1 region ({e0e1_center:.6g}) sits nearer the ZERO-flux family "
                f"than the half-flux one; the e0-e1 branch is only populated where "
                f"f01 is minimum, so this contradicts the parity call")
        elif e0e1_agree > e0e1_disagree_max:
            degrade = True
            notes.append(f"e0-e1 region is {e0e1_agree:.3f} P from the predicted "
                         f"half flux (>{e0e1_disagree_max:.2f} P)")
    if not n_two:
        notes.append("no two-notch traces, so parity rests on the CNN alone")

    # -- status ---------------------------------------------------------------
    confident = max(pred["parity_prob"], 1 - pred["parity_prob"]) >= parity_conf_min
    contradicted = any("contradicts the parity call" in n for n in notes)

    if contradicted and not confident:
        notes.append("parity is neither confident nor corroborated")
        return _finish(FluxPoints(STATUS_INCONCLUSIVE, period=period,
                                  period_source=period_source, symmetry=q,
                                  symmetry_result=sym,
                                  notes=notes + q.reasons), verbose)
    if not q.usable and not confident:
        notes.append("symmetry failed and the CNN is not confident either")
        return _finish(FluxPoints(STATUS_INCONCLUSIVE, period=period,
                                  period_source=period_source, symmetry=q,
                                  symmetry_result=sym,
                                  notes=notes + q.reasons), verbose)

    status = (STATUS_UNCERTAIN
              if (degrade or contradicted or not confident or q.status != SYM_OK)
              else STATUS_OK)

    out = FluxPoints(
        status=status,
        zero_flux=zero_anchor, half_flux=half_anchor,
        zero_all=zero_all, half_all=half_all,
        period=period, period_source=period_source,
        position_source=position_source,
        parity_prob=pred["parity_prob"], parity_spread=pred["parity_spread"],
        e0e1_center=e0e1_center, e0e1_agreement=e0e1_agree,
        symmetry=q, symmetry_result=sym, notes=notes + q.reasons,
    )
    return _finish(out, verbose)
