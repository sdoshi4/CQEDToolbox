"""Flux symmetry search for resonator-vs-flux sweeps.

Vendored from the `Fluxonium-offset-inverse-model` repository (`flux_symmetry.py`,
itself extracted verbatim from `01_find_flux_points.ipynb`).  The algorithms are
unchanged; only the notebook-facing half was dropped -- `load_sweep`, `describe`,
`report`, `analyze`, `plot_all`, `compare` and the `DATASETS` registry, all of
which exist to drive the notebook against files on disk.  Callers here build the
`xr.Dataset` in memory from an operation that has already loaded its data.

What the search does, and why it is the half that owns the *positions*: a
resonator-vs-flux curve is even about both zero and half flux, so both are mirror
centres of the curve.  Folding the curve about a trial centre and scoring the
mismatch locates those centres to 0.3-5.1 uA on real sweeps, against 13.7-15.8 uA
for the CNN.  What it cannot do is say *which* family is zero flux and which is
half: no geometric test can, since the two families are internally identical.
That bit comes from the CNN's parity head (see `cnn.py`).
"""

from dataclasses import dataclass, field
from typing import Optional, List
from itertools import combinations

import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit


# ----------------------------------------------------------------------------
# Resonance extraction: affine background + one or two Lorentzians, BIC-selected.
#
# This beats fitting the complex DoubleHangerResponseBruno on every row.  That
# model has 10 parameters against ~2 points per linewidth here, and on
# low-contrast rows it escapes to a delta (kappa -> 0) or to a baseline
# (kappa ~ 10 MHz against a real 0.20 MHz).  The bounded-HWHM + BIC formulation
# below has neither failure mode, and it localises the second resonance to the
# half-flux region instead of scattering it across the whole sweep.
#
# Note this is a *second* extractor, distinct from `fit_lorentzian_notches` in
# res_spec_vs_flux.py.  Both are needed: this one reports per-resonance
# amplitudes, which is the only way `dominant_series` can pick the g0-g1 branch,
# while `fit_lorentzian_notches` supplies the sorted two-branch channels the CNN
# consumes.
# ----------------------------------------------------------------------------

def extract_resonator_signal(
    sweep,
    signal: str = "signal",
    *,
    current_dim: str = "current",
    frequency_dim: str = "freq",
    polarity: str = "dip",
    smooth_sigma: float = 1.5,
    min_prominence: float = 0.15,
    min_separation_bins: int = 3,
    max_candidates: int = 4,
    double_bic_improvement: float = 6.0,
    max_hwhm_fraction: float = 0.10,
) -> pd.DataFrame:
    """Extract one or two Lorentzian resonances from each current trace.

    Each trace is fitted with an affine background plus either one or two
    Lorentzians.  Candidate centres are first located in a smoothed trace;
    all plausible two-centre combinations are then fitted.  The double model
    is kept only if its Bayesian information criterion (BIC) improves on the
    best single-resonance fit by ``double_bic_improvement`` or more.

    Parameters
    ----------
    sweep
        xarray dataset containing a 2-D magnitude (or power) trace.
    signal
        Name of the magnitude/power data variable.
    polarity
        ``"dip"`` for hanger notches (the default), or ``"peak"`` for
        spectra with upward-going Lorentzians.
    smooth_sigma
        Gaussian smoothing width in frequency samples, used only to seed the
        fits.  The fit itself always uses the unsmoothed trace.
    min_prominence
        Minimum candidate prominence, as a fraction of the trace range.
    min_separation_bins
        Minimum spacing between two resolved resonances in frequency samples.
    double_bic_improvement
        Required BIC reduction before reporting two resonances.  A value of 6
        is commonly interpreted as strong evidence for the extra component.
    max_hwhm_fraction
        Maximum fitted half-width at half maximum, relative to the full
        frequency span.  It prevents a broad Lorentzian from modelling a
        slowly varying background.

    Returns
    -------
    pd.DataFrame
        One row per accepted resonance.  ``resonance_index`` orders the
        fitted centres from low to high frequency within each current trace.
        ``amplitude`` is positive for both dips and peaks, and ``fwhm`` is in
        the same frequency unit as ``sweep[frequency_dim]``.
    """
    if signal not in sweep:
        raise KeyError(f"{signal!r} is not a variable in the supplied sweep")
    if polarity not in {"dip", "peak"}:
        raise ValueError("polarity must be either 'dip' or 'peak'")
    if smooth_sigma <= 0 or min_prominence <= 0 or max_hwhm_fraction <= 0:
        raise ValueError("smooth_sigma, min_prominence, and max_hwhm_fraction must be positive")
    if min_separation_bins < 1 or max_candidates < 1:
        raise ValueError("min_separation_bins and max_candidates must be at least one")

    trace_da = sweep[signal].transpose(current_dim, frequency_dim)
    if trace_da.ndim != 2:
        raise ValueError(f"{signal!r} must have exactly the dimensions ({current_dim!r}, {frequency_dim!r})")

    currents = np.asarray(trace_da.coords[current_dim].values)
    frequencies = np.asarray(trace_da.coords[frequency_dim].values, dtype=float)
    traces = np.asarray(trace_da.values, dtype=float)
    if frequencies.ndim != 1:
        raise ValueError("This extractor requires one shared 1-D frequency axis")

    # Positive amplitudes always represent the magnitude of the feature.
    feature_sign = -1.0 if polarity == "dip" else 1.0
    output_columns = [
        "current", "frequency", "resonance_index", "n_resonances",
        "amplitude", "fwhm", "fit_rms", "model_bic",
        "bic_gain_for_double", "candidate_prominence",
    ]
    records = []

    for current, raw_trace in zip(currents, traces):
        finite = np.isfinite(frequencies) & np.isfinite(raw_trace)
        if finite.sum() < 15:
            continue

        frequency = frequencies[finite]
        trace = raw_trace[finite]
        order = np.argsort(frequency)
        frequency, trace = frequency[order], trace[order]
        if np.any(np.diff(frequency) <= 0):
            # Duplicate frequency points make widths ill-defined.
            continue

        trace_scale = np.ptp(trace)
        if not np.isfinite(trace_scale) or trace_scale <= np.finfo(float).eps:
            continue

        frequency_mid = 0.5 * (frequency[0] + frequency[-1])
        frequency_half_span = 0.5 * (frequency[-1] - frequency[0])
        x = (frequency - frequency_mid) / frequency_half_span
        y = (trace - np.median(trace)) / trace_scale
        x_step = float(np.median(np.diff(x)))

        # The outer 1/8 of the trace estimates the cable/background slope.
        n_edge = min(30, max(5, frequency.size // 8))
        edge_indices = np.r_[np.arange(n_edge), np.arange(frequency.size - n_edge, frequency.size)]
        background_slope, background_offset = np.polyfit(x[edge_indices], y[edge_indices], 1)
        background = background_offset + background_slope * x

        smoothed = gaussian_filter1d(y, sigma=smooth_sigma, mode="nearest")
        feature = feature_sign * (smoothed - background)
        distance = max(int(min_separation_bins), int(np.ceil(2 * smooth_sigma)))
        peaks, peak_properties = find_peaks(
            feature,
            prominence=min_prominence * np.ptp(smoothed),
            distance=distance,
        )
        if not len(peaks):
            continue

        prominence_by_index = {
            int(index): float(prominence)
            for index, prominence in zip(peaks, peak_properties["prominences"])
        }
        candidate_indices = sorted(
            prominence_by_index, key=prominence_by_index.get, reverse=True
        )[:max_candidates]

        min_hwhm = 0.5 * x_step
        max_hwhm = max_hwhm_fraction * (frequency[-1] - frequency[0]) / frequency_half_span
        if max_hwhm <= min_hwhm:
            raise ValueError("max_hwhm_fraction is too small for this frequency sampling")
        # A valid fit may refine a peak by several bins, but cannot migrate to
        # a different unrelated fluctuation in the trace.
        max_centre_shift = max(6 * x_step, 0.03 * (frequency[-1] - frequency[0]) / frequency_half_span)

        def lorentzian_model(x_values, *parameters):
            offset, slope = parameters[:2]
            result = offset + slope * x_values
            for amplitude, centre, hwhm in np.asarray(parameters[2:]).reshape(-1, 3):
                result += feature_sign * amplitude / (1 + ((x_values - centre) / hwhm) ** 2)
            return result

        def fit_model(initial_indices):
            """Fit one candidate set and reject unstable/relocated fits."""
            n_components = len(initial_indices)
            widths_in_bins = peak_widths(feature, initial_indices, rel_height=0.5)[0]
            initial_parameters = [background_offset, background_slope]
            for index, width_in_bins in zip(initial_indices, widths_in_bins):
                initial_parameters.extend([
                    max(feature[index], 0.02),
                    x[index],
                    np.clip(0.5 * width_in_bins * x_step, min_hwhm, 0.8 * max_hwhm),
                ])

            lower_bounds = [-5, -5] + sum(([0, -1, min_hwhm] for _ in range(n_components)), [])
            upper_bounds = [5, 5] + sum(([5, 1, max_hwhm] for _ in range(n_components)), [])
            try:
                parameters, _ = curve_fit(
                    lorentzian_model, x, y, p0=initial_parameters,
                    bounds=(lower_bounds, upper_bounds), maxfev=20_000,
                )
            except (RuntimeError, ValueError, FloatingPointError):
                return None

            components = np.asarray(parameters[2:]).reshape(-1, 3)
            initial_centres = x[np.asarray(initial_indices)]
            if np.any(np.abs(components[:, 1] - initial_centres) > max_centre_shift):
                return None
            if np.any(components[:, 2] >= 0.995 * max_hwhm):
                return None
            if n_components == 2 and abs(components[0, 1] - components[1, 1]) < min_separation_bins * x_step:
                return None

            residuals = y - lorentzian_model(x, *parameters)
            rss = float(np.sum(residuals**2))
            bic = frequency.size * np.log(max(rss / frequency.size, np.finfo(float).eps))
            bic += parameters.size * np.log(frequency.size)
            return {
                "bic": bic,
                "rss": rss,
                "components": components,
                "prominence": [prominence_by_index[index] for index in initial_indices],
            }

        single_fits = [fit_model([index]) for index in candidate_indices]
        single_fits = [fit for fit in single_fits if fit is not None]
        if not single_fits:
            continue
        best_single = min(single_fits, key=lambda fit: fit["bic"])

        double_fits = [
            fit_model(indices)
            for indices in combinations(candidate_indices, 2)
        ]
        double_fits = [fit for fit in double_fits if fit is not None]
        best_double = min(double_fits, key=lambda fit: fit["bic"]) if double_fits else None
        bic_gain_for_double = (
            best_single["bic"] - best_double["bic"]
            if best_double is not None else 0.0
        )
        selected = (
            best_double
            if best_double is not None and bic_gain_for_double >= double_bic_improvement
            else best_single
        )

        component_rows = sorted(
            zip(selected["components"], selected["prominence"]),
            key=lambda item: item[0][1],
        )
        fit_rms = np.sqrt(selected["rss"] / frequency.size) * trace_scale
        for resonance_index, ((amplitude, centre, hwhm), prominence) in enumerate(component_rows):
            records.append({
                "current": current,
                "frequency": frequency_mid + centre * frequency_half_span,
                "resonance_index": resonance_index,
                "n_resonances": len(component_rows),
                "amplitude": amplitude * trace_scale,
                "fwhm": 2 * hwhm * frequency_half_span,
                "fit_rms": fit_rms,
                "model_bic": selected["bic"],
                "bic_gain_for_double": bic_gain_for_double,
                "candidate_prominence": prominence * trace_scale,
            })

    return pd.DataFrame.from_records(records, columns=output_columns)


def filter_outliers(flux, y, valid, window: int = 16, mad_mult: float = 20.0):
    """Drop points far from a running median of y.  Operates on one series only.

    Deliberately does *not* consider whether a second branch was found: the old
    robust_filter_fitted_curve rejected exactly the rows near half flux where the
    fit legitimately moves onto another branch.
    """
    y = np.asarray(y, float)
    keep = np.asarray(valid, bool).copy()
    tmp = np.where(keep, y, np.nan)

    med = np.full(len(y), np.nan)
    half = window // 2
    for i in range(len(y)):
        seg = tmp[max(0, i - half): i + half + 1]
        if np.any(np.isfinite(seg)):
            med[i] = np.nanmedian(seg)

    dev = np.abs(y - med)
    mad = np.nanmedian(dev[keep]) if keep.any() else np.nan
    if not np.isfinite(mad) or mad <= 0:
        return keep, med
    return keep & (dev <= mad_mult * mad), med


@dataclass
class SymmetryResult:
    grid: np.ndarray                 # uniform current grid
    residual: np.ndarray             # R(c), nan where not evaluable
    centers: np.ndarray              # refined symmetry-centre positions
    center_res: np.ndarray           # R at each centre
    center_width: np.ndarray         # curvature width (see _refine)
    period: Optional[float]
    period_method: str
    half_width: float
    notes: List[str] = field(default_factory=list)
    boot_period: Optional[float] = None      # from the translational scan
    trial_periods: Optional[np.ndarray] = None
    trial_residual: Optional[np.ndarray] = None
    lattice_rms: Optional[float] = None      # rms of the centres about the lattice
    lattice_indices: Optional[List[int]] = None   # integer site of each centre


def _to_masked_grid(x, y, valid, n_grid: int = 2000, max_gap_mult: float = 3.0):
    """Resample onto a uniform grid, marking grid points near a real sample.

    Gaps wider than ``max_gap_mult`` x the median sample spacing are marked invalid
    instead of being interpolated across -- the failure that made the baseline's
    FFT autocorrelation return 225.7 and 614.6 for the same qubit.
    """
    x, y, valid = np.asarray(x, float), np.asarray(y, float), np.asarray(valid, bool)
    xs, ys = x[valid], y[valid]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if xs.size < 4:
        raise ValueError("not enough valid points for a symmetry search")

    grid = np.linspace(xs.min(), xs.max(), n_grid)
    vals = np.interp(grid, xs, ys)

    step = np.median(np.diff(np.sort(x)))
    tol = max_gap_mult * step
    nearest = np.abs(grid[:, None] - xs[None, :]).min(axis=1)
    ok = nearest <= tol
    return grid, vals, ok


def fold_residual(grid, vals, ok, half_width: float, n_delta: int = 240,
                  min_valid_frac: float = 0.5):
    """R(c) = mean_delta [f(c+d) - f(c-d)]^2 / Var(f in the window).

    Only deltas where *both* c+d and c-d are valid contribute.  The variance
    normalisation matters: without it any flat stretch of the curve is trivially
    symmetric and scores as a perfect centre.
    """
    deltas = np.linspace(0.0, half_width, n_delta)
    dx = grid[1] - grid[0]
    res = np.full(grid.size, np.nan)
    var_out = np.full(grid.size, np.nan)

    for i, c in enumerate(grid):
        ip = np.rint((c + deltas - grid[0]) / dx).astype(int)
        im = np.rint((c - deltas - grid[0]) / dx).astype(int)
        inside = (ip >= 0) & (ip < grid.size) & (im >= 0) & (im < grid.size)
        if not inside.any():
            continue
        ip, im = ip[inside], im[inside]
        good = ok[ip] & ok[im]
        if good.sum() < min_valid_frac * n_delta:
            continue
        a, b = vals[ip[good]], vals[im[good]]
        var = np.var(np.concatenate([a, b]))
        if var <= 0:
            continue
        res[i] = np.mean((a - b) ** 2) / var
        var_out[i] = var
    return res, var_out


def _noise_sigma(x, y, valid) -> float:
    """Robust point-to-point noise of y, via second differences.

    A running median is useless here: over a window the signal is monotonic, the
    median of the window *is* the centre sample, so the deviation is identically
    zero and sigma comes out 0.  Second differences annihilate any locally linear
    trend and leave the noise, with Var(y[i-1] - 2y[i] + y[i+1]) = 6 sigma^2.
    """
    v = np.asarray(valid, bool)
    x, y = np.asarray(x, float)[v], np.asarray(y, float)[v]
    order = np.argsort(x)
    x, y = x[order], y[order]
    if y.size < 8:
        return float("nan")

    d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
    # only trust triples that are close to evenly spaced
    ga, gb = x[1:-1] - x[:-2], x[2:] - x[1:-1]
    even = np.abs(ga - gb) <= 0.5 * np.minimum(ga, gb)
    d2 = np.abs(d2[even & np.isfinite(d2)])
    if d2.size < 5:
        return float("nan")
    # Fitted frequencies often repeat exactly (the fit pins to the same value on
    # many rows), which drives the median to 0.  Fall back to the spread of the
    # values that do vary.
    s = 1.4826 * np.median(d2) / np.sqrt(6.0)
    if s <= 0:
        nz = d2[d2 > 0]
        s = (1.4826 * np.median(nz) / np.sqrt(6.0)) if nz.size >= 5 else float("nan")
    return float(s)


def _fit_lattice(centers, boot_period=None, tol: float = 0.05):
    """Coarsest half-period h such that every centre sits on the lattice c0 + n*h.

    Taking the period straight from "centres 1 and 3" assumes no centre in the
    run was missed.  On the June sweep exactly that happens -- the centres come
    out at 6.69e-5, 1.61e-4, 3.47e-4, so c3-c1 spans *two* periods and gives 2.8e-4
    against a true 1.87e-4.  Fitting a lattice instead assigns integer indices, so
    a gap in the run costs nothing.

    Returns (half_period, indices, rms_residual) or (None, None, None).
    """
    centers = np.asarray(centers, float)
    if centers.size < 2:
        return None, None, None

    cands = []
    for d in np.diff(centers):
        cands += [d / k for k in (1, 2, 3, 4)]
    if boot_period:
        cands.append(boot_period / 2.0)
    cands = sorted({round(h, 12) for h in cands if h > 0}, reverse=True)

    # The independent translational estimate outranks "coarsest lattice that
    # fits".  With a centre missed, two surviving centres a full period apart fit
    # h = P perfectly with indices [0, 1] and would report twice the true period
    # (183057: 270.6 and 904.4 give 1268 against a true 634).  Restricting to
    # half-periods near boot/2 assigns indices [0, 2] instead.
    if boot_period:
        near = [h for h in cands if abs(h - boot_period / 2.0) <= 0.15 * boot_period / 2.0]
        if near:
            cands = near

    for h in cands:                       # coarsest lattice that still fits
        n = np.round((centers - centers[0]) / h)
        if np.unique(n).size != n.size:   # two centres on one lattice site
            continue
        A = np.vstack([np.ones_like(n), n]).T
        coef, *_ = np.linalg.lstsq(A, centers, rcond=None)
        if coef[1] <= 0:
            continue
        rms = float(np.sqrt(np.mean((centers - A @ coef) ** 2)))
        if rms <= tol * coef[1]:
            return float(coef[1]), n.astype(int), rms
    return None, None, None


def translational_residual(grid, vals, ok, periods, min_overlap_frac: float = 0.25):
    """T(P) = mean_I [f(I+P) - f(I)]^2 / Var, over the masked overlap.

    Used only to bootstrap the fold half-width, since HW should be ~P/4 but P is
    what we are after.  Unlike an FFT autocorrelation this never interpolates
    across a gap, and the variance normalisation makes T comparable across P.
    A curve even about both zero and half flux does *not* repeat at P/2, so the
    first deep minimum is the true period.
    """
    dx = grid[1] - grid[0]
    n = grid.size
    out = np.full(len(periods), np.nan)
    var_out = np.full(len(periods), np.nan)
    for k, p in enumerate(periods):
        shift = int(round(p / dx))
        if shift <= 0 or shift >= n:
            continue
        a, b = vals[shift:], vals[:n - shift]
        good = ok[shift:] & ok[:n - shift]
        if good.sum() < min_overlap_frac * (n - shift) or good.sum() < 20:
            continue
        a, b = a[good], b[good]
        var = np.var(np.concatenate([a, b]))
        if var <= 0:
            continue
        out[k] = np.mean((a - b) ** 2) / var
        var_out[k] = var
    return out, var_out


def bootstrap_period(grid, vals, ok, sigma: float = float("nan"),
                     min_frac: float = 0.10, max_frac: float = 0.90,
                     threshold: float = 0.25, n_trial: int = 600):
    """Smallest period whose translational residual is a deep local minimum."""
    span = grid[-1] - grid[0]
    periods = np.linspace(min_frac * span, max_frac * span, n_trial)
    t, _tvar = translational_residual(grid, vals, ok, periods)
    finite = np.isfinite(t)
    if finite.sum() < 10:
        return None, periods, t

    filled = np.where(finite, t, np.nanmax(t[finite]))
    idx, _ = find_peaks(-filled, prominence=0.05)
    idx = idx[filled[idx] < threshold]
    edge = 0.02 * (periods[-1] - periods[0])
    on_edge = lambda p: (p - periods[0] < edge) or (periods[-1] - p < edge)
    if idx.size == 0:
        best = int(np.nanargmin(filled))
        p = float(periods[best])
        return (p if (filled[best] < threshold and not on_edge(p)) else None), periods, t
    for j in idx:                      # first interior minimum
        p = float(periods[j])
        if not on_edge(p):
            return p, periods, t
    return None, periods, t


def _refine(grid, res, idx):
    """Sub-grid centre by parabolic interpolation; width where R rises to 2*Rmin."""
    if idx <= 0 or idx >= len(res) - 1:
        return grid[idx], res[idx], np.nan
    y0, y1, y2 = res[idx - 1], res[idx], res[idx + 1]
    denom = y0 - 2 * y1 + y2
    if not np.isfinite(denom) or denom == 0:
        return grid[idx], y1, np.nan
    dx = grid[1] - grid[0]
    shift = 0.5 * (y0 - y2) / denom
    c = grid[idx] + shift * dx
    rmin = y1 - 0.25 * (y0 - y2) * shift
    curv = denom / dx ** 2                      # R ~ rmin + 0.5*curv*(x-c)^2
    width = np.sqrt(2 * rmin / curv) if (curv > 0 and rmin > 0) else np.nan
    return c, rmin, width


def find_centers(grid, res, min_separation: float, threshold: float = 0.10,
                 prominence_frac: float = 0.15):
    """Local minima of R that are both prominent and genuinely deep.

    ``threshold`` is absolute, which is only meaningful because R is normalised by
    the in-window variance: R ~ 2 for two uncorrelated halves, R -> 0 for a true
    mirror centre.  R < 0.1 means the fold mismatch is under 10% of the curve
    variance.  A relative cut (e.g. a fraction of the median R) does not work --
    real centres sit 3-4 orders of magnitude below the spurious ones, so a cut
    scaled to the median lets shallow false minima through.
    """
    finite = np.isfinite(res)
    if finite.sum() < 10:
        return np.array([]), np.array([]), np.array([])

    filled = np.where(finite, res, np.nanmax(res[finite]))
    spread = np.nanmedian(filled) - np.nanmin(filled)
    dx = grid[1] - grid[0]
    idx, _ = find_peaks(-filled,
                        distance=max(1, int(min_separation / dx)),
                        prominence=max(prominence_frac * spread, 1e-12))
    if idx.size == 0:
        return np.array([]), np.array([]), np.array([])

    idx = idx[filled[idx] < threshold]
    refined = [_refine(grid, filled, int(i)) for i in idx]
    if not refined:
        return np.array([]), np.array([]), np.array([])
    c, r, w = map(np.asarray, zip(*refined))
    order = np.argsort(c)
    return c[order], r[order], w[order]


def find_symmetry(flux, y, valid, n_grid: int = 2000, threshold: float = 0.10,
                  half_width: Optional[float] = None,
                  verbose: bool = True) -> SymmetryResult:
    """Bootstrap the period by masked translation, then fold at HW = P/4."""
    grid, vals, ok = _to_masked_grid(flux, y, valid, n_grid=n_grid)
    span = grid[-1] - grid[0]
    notes: List[str] = []
    sigma = _noise_sigma(flux, y, valid)

    # --- bootstrap the half-width ---------------------------------------------
    boot, periods, tres = bootstrap_period(grid, vals, ok, sigma=sigma)
    if verbose:
        print(f"  translational period estimate: "
              f"{'none' if boot is None else f'{boot:.4g}'}")
    if half_width is not None:
        hw = float(half_width)
    elif boot is not None:
        hw = boot / 4.0
    else:
        hw = 0.10 * span
        notes.append("no translational period found; folding at 10% of the span")

    # --- fold ------------------------------------------------------------------
    res, _var = fold_residual(grid, vals, ok, hw)
    centers, cres, cwidth = find_centers(grid, res, min_separation=hw,
                                         threshold=threshold)
    if verbose:
        print(f"  noise sigma={sigma:.4g}; fold (HW={hw:.4g}): {len(centers)} "
              f"centre(s) at {np.array2string(centers, precision=4)}")

    if len(centers) < 2:
        notes.append(f"fewer than 2 symmetry centres found at HW={hw:.4g}")
        return SymmetryResult(grid, res, centers, cres, cwidth,
                              boot, "translational only (too few centres)", hw, notes,
                              boot, periods, tres)

    res2, hw2 = res, hw

    # --- period ----------------------------------------------------------------
    half, idx, rms = _fit_lattice(centers, boot)
    if half is None:
        period, method = None, "centres do not fall on a common lattice"
        notes.append("the detected centres are not consistent with one period; "
                     "treat this dataset as a failure rather than a result")
    elif idx is not None and int(idx.max()) >= 2:
        period = 2.0 * half
        method = (f"lattice fit over {len(centers)} centres, "
                  f"indices {[int(v) for v in idx]}")
        if int(idx.max()) + 1 > len(centers):
            notes.append(f"a centre was missed inside the run "
                         f"(indices {[int(v) for v in idx]}); the lattice fit "
                         "spans a full period and accounts for it")
    else:
        period = 2.0 * half
        method = "2 x adjacent spacing (UNVERIFIED - a centre between them would be missed)"
        notes.append(
            "Only 2 symmetry centres are inside the foldable range, so the period is "
            "2 x their spacing. Extend the sweep by ~1 half-period so a third centre "
            "is foldable and the period can be cross-checked against the lattice.")

    foldable = (grid[0] + hw2, grid[-1] - hw2)
    notes.append(f"foldable range at HW={hw2:.4g}: {foldable[0]:.4g} .. {foldable[1]:.4g}")
    if period is not None:
        nxt = centers[-1] + period / 2
        if nxt > foldable[1]:
            notes.append(
                f"the next centre is predicted near {nxt:.4g}, outside the foldable "
                f"range; data out to ~{nxt + hw2:.4g} would capture it")
    # Independent cross-check: the centres and the translational scan should agree.
    if period is not None and boot is not None:
        if abs(period - boot) > 0.10 * period:
            notes.append(
                f"WARNING: period from centres ({period:.4g}) disagrees with the "
                f"translational estimate ({boot:.4g}) by more than 10% -- a centre "
                f"was probably missed or a spurious one accepted")

    return SymmetryResult(grid, res2, centers, cres, cwidth, period, method, hw2,
                          notes, boot, periods, tres,
                          lattice_rms=rms,
                          lattice_indices=None if idx is None else [int(v) for v in idx])


# ----------------------------------------------------------------------------
# Quality gate: can the symmetry result be trusted?
# ----------------------------------------------------------------------------

SYM_OK = "ok"
SYM_UNCERTAIN = "uncertain"
SYM_FAILED = "failed"


@dataclass
class SymmetryQuality:
    """Verdict on a SymmetryResult, plus the numbers behind it."""
    status: str                              # SYM_OK / SYM_UNCERTAIN / SYM_FAILED
    reasons: List[str] = field(default_factory=list)
    n_centers: int = 0
    period_disagreement: Optional[float] = None   # |lattice - translational| / lattice
    max_center_res: Optional[float] = None
    lattice_indices: Optional[List[int]] = None

    @property
    def usable(self) -> bool:
        return self.status in (SYM_OK, SYM_UNCERTAIN)


def symmetry_quality(sym: SymmetryResult,
                     fail_disagreement: float = 0.10,
                     warn_disagreement: float = 0.02,
                     warn_center_res: float = 0.095) -> SymmetryQuality:
    """Decide whether the symmetry search can be trusted for flux-point positions.

    The discriminator is the agreement between the two *independent* period
    estimates the search already produces: the lattice fitted through the centre
    positions, and the masked translational scan over the whole curve.  On the
    four real sweeps these agree to 0.03-0.29%, so the 10% failure threshold has
    an enormous margin.

    The fold residual deliberately is not the primary signal, and is weak:
    `find_centers` already rejects anything above 0.10, so every surviving centre
    is below that by construction, and perfectly good sweeps reach 0.081.  The
    warn threshold therefore sits just under the hard filter -- set at 0.08 it
    fired on aug11-183057, which is one of the best results we have.

    Lattice indices of exactly [0, 1] earn a downgrade because the period is then
    twice a single centre spacing -- a centre missed in between would double it.
    Indices like [0, 2] already span a full period and are safe.
    """
    reasons: List[str] = []
    n = int(len(sym.centers))
    disagreement = None
    max_res = float(np.max(sym.center_res)) if len(sym.center_res) else None

    if n < 2:
        reasons.append(f"only {n} symmetry centre(s) found; need at least 2")
        return SymmetryQuality(SYM_FAILED, reasons, n, None, max_res,
                               sym.lattice_indices)

    if sym.period is None:
        reasons.append("centres do not fall on a common lattice, so no period")
        return SymmetryQuality(SYM_FAILED, reasons, n, None, max_res,
                               sym.lattice_indices)

    if sym.boot_period:
        disagreement = abs(sym.period - sym.boot_period) / abs(sym.period)
        if disagreement > fail_disagreement:
            reasons.append(
                f"lattice period {sym.period:.6g} and translational period "
                f"{sym.boot_period:.6g} disagree by {100*disagreement:.1f}% "
                f"(>{100*fail_disagreement:.0f}%)")
            return SymmetryQuality(SYM_FAILED, reasons, n, disagreement, max_res,
                                   sym.lattice_indices)
        if disagreement > warn_disagreement:
            reasons.append(
                f"period estimates differ by {100*disagreement:.1f}% "
                f"(>{100*warn_disagreement:.0f}%)")
    else:
        reasons.append("no translational period, so the lattice is unchecked")

    if max_res is not None and max_res > warn_center_res:
        reasons.append(f"worst fold residual {max_res:.4g} is high "
                       f"(>{warn_center_res:.2g})")

    if sym.lattice_indices == [0, 1]:
        reasons.append("period is 2x a single centre spacing; a centre missed "
                       "between them would double it")

    status = SYM_UNCERTAIN if reasons else SYM_OK
    return SymmetryQuality(status, reasons, n, disagreement, max_res,
                           sym.lattice_indices)


def dominant_series(resonances: pd.DataFrame):
    """One frequency per current: the resonance carrying the largest amplitude.

    Where a trace has two resonances this is the deeper one, i.e. the g0-g1 line
    (the e0-e1 partner only appears near half flux and is always the weaker of
    the pair).  Folding a *derived* quantity instead -- the low branch, the
    weighted centroid, the "dominant" branch by mixture weight -- was measurably
    worse: those flip or drift wherever the second component is spurious, and the
    fold contrast against the off-centre level drops from ~100x to ~3x.
    """
    dom = resonances.loc[resonances.groupby("current")["amplitude"].idxmax()]
    dom = dom.sort_values("current")
    return dom["current"].to_numpy(float), dom["frequency"].to_numpy(float)
