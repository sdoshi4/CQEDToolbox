"""Zero- and half-flux points from a resonator-vs-flux sweep.

A fold-symmetry search owns the *positions* and the *period*; a small CNN
ensemble owns only the *parity* bit -- which of the two geometrically identical
families of symmetry centres is zero flux.  See `pipeline.locate_flux_points`.

Vendored from the `Fluxonium-offset-inverse-model` repository; training lives
there and is deliberately not reproduced here.  Importing this package needs
only numpy/scipy/pandas -- `torch` is pulled in lazily by `load_model` and
`predict`, so `cqedtoolbox` stays importable without the `ml` extra.
"""

from .pipeline import (
    FluxPoints, locate_flux_points, extract_branches,
    STATUS_OK, STATUS_UNCERTAIN, STATUS_INCONCLUSIVE, CNN_KNOWN_BIAS_PERIODS,
)
from .cnn import load_model, predict, N_POINTS, N_CHANNELS
from .symmetry import (
    SymmetryResult, SymmetryQuality, find_symmetry, symmetry_quality,
    extract_resonator_signal, dominant_series, filter_outliers,
    SYM_OK, SYM_UNCERTAIN, SYM_FAILED,
)

__all__ = [
    "FluxPoints", "locate_flux_points", "extract_branches",
    "STATUS_OK", "STATUS_UNCERTAIN", "STATUS_INCONCLUSIVE",
    "CNN_KNOWN_BIAS_PERIODS",
    "load_model", "predict", "N_POINTS", "N_CHANNELS",
    "SymmetryResult", "SymmetryQuality", "find_symmetry", "symmetry_quality",
    "extract_resonator_signal", "dominant_series", "filter_outliers",
    "SYM_OK", "SYM_UNCERTAIN", "SYM_FAILED",
]
