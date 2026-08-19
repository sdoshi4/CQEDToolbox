"""Forward model and preprocessing for the fluxonium flux-point CNN.

Vendored from the `Fluxonium-offset-inverse-model` repository (`flux_ml.py`),
keeping only the *inference* half: the channel builder, the label decoder, the
network definition, and checkpoint loading.  The simulation and training half
(`simulate_period`, `Acquisition`, `observable_branches`, `Spans`,
`make_examples`, `generate_dataset`, `train_model`, ...) stays in that
repository, since retraining is not something a protocol run does.  Dropping it
also drops the module-level `import scqubits`.

`torch` is imported lazily through `_torch()` so this module -- and the whole
operations package -- stays importable on machines without it.

The channel builder here must stay byte-for-byte equivalent to the one used at
training time.  The two paths building channels differently is a real bug that
has happened: two resamplers broke ties differently and put the paths one source
sample apart, which is 1/256 of a period and the same size as the model's own
error.  Cell 6 of `03_predict_flux_points.ipynb` asserts they still agree.
"""

from typing import Dict, Any, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# 1. Channel builder -- the one path shared by training and inference
# ----------------------------------------------------------------------------

N_POINTS = 128          # resampled points per flux period
N_CHANNELS = 4          # lo, hi, has_second, valid


def build_channels(f_low, f_high, has_second, valid) -> np.ndarray:
    """Stack the CNN input, (N_CHANNELS, N) float32.

    Frequencies are converted to MHz and centred on the median of the valid `f_low`
    """
    f_low = np.asarray(f_low, float)
    f_high = np.asarray(f_high, float)
    has_second = np.asarray(has_second, bool)
    valid = np.asarray(valid, bool)

    ref = np.median(f_low[valid]) if valid.any() else 0.0
    lo = np.where(valid, (f_low - ref) * 1e3, 0.0)
    hi = np.where(valid, (f_high - ref) * 1e3, 0.0)

    return np.stack([lo, hi,
                     (has_second & valid).astype(float),
                     valid.astype(float)]).astype(np.float32)


def resample_period(x, f_low, f_high, has_second, valid, x_start, period,
                    n_points: int = N_POINTS):
    """Sample one period starting at ``x_start`` onto a uniform grid.

    ``x`` is the flux/current axis of the input series.  Nearest-sample
    assignment (not interpolation) is used so that gaps stay gaps: interpolating
    across a hole is precisely what made the old autocorrelation period estimate
    fabricate structure.  A grid point with no sample within half a grid step is
    marked invalid.
    """
    x = np.asarray(x, float)
    valid = np.asarray(valid, bool)
    grid = x_start + (np.arange(n_points) + 0.5) * period / n_points

    xs = x[valid]
    if xs.size == 0:
        z = np.zeros(n_points)
        return z, z, np.zeros(n_points, bool), np.zeros(n_points, bool)

    idx_valid = np.where(valid)[0]
    j = np.searchsorted(xs, grid).clip(1, xs.size - 1)
    pick = np.where(np.abs(grid - xs[j - 1]) <= np.abs(grid - xs[j]), j - 1, j)
    src = idx_valid[pick]

    tol = 0.5 * period / n_points + 0.5 * np.median(np.diff(xs)) if xs.size > 1 \
        else 0.5 * period / n_points
    ok = np.abs(grid - xs[pick]) <= tol

    return (np.asarray(f_low, float)[src],
            np.asarray(f_high, float)[src],
            np.asarray(has_second, bool)[src] & ok,
            ok)


def channels_from_series(x, f_low, f_high, has_second, valid, x_start, period,
                         n_points: int = N_POINTS) -> np.ndarray:
    """Crop one period from an irregular series and build the CNN input."""
    lo, hi, sec, ok = resample_period(x, f_low, f_high, has_second, valid,
                                      x_start, period, n_points)
    return build_channels(lo, hi, sec, ok)


# ----------------------------------------------------------------------------
# 2. Labels
# ----------------------------------------------------------------------------

def encode_labels(zero_offset: float) -> Tuple[np.ndarray, float]:
    """Split the zero-flux offset into the two heads.

    ``zero_offset`` is where zero flux sits inside the cropped window, as a
    fraction of the period.

    Head A regresses ``(cos, sin)`` of ``2*pi * 2 * offset`` -- the position of
    the nearest symmetry point, which repeats every *half* period.  Head B is a
    single bit: is that symmetry point zero flux (1) or half flux (0)?

    Only training calls this, but it is kept as the written record of the
    convention that `decode_labels` inverts.
    """
    frac = float(zero_offset) % 1.0
    theta = 2.0 * np.pi * (2.0 * frac)
    parity = 1.0 if frac < 0.5 else 0.0
    return np.array([np.cos(theta), np.sin(theta)], dtype=np.float32), parity


def decode_labels(cos_sin, parity_prob: float) -> Dict[str, float]:
    """Invert `encode_labels`; returns offsets as fractions of the period."""
    c, s = np.asarray(cos_sin, float)
    n = np.hypot(c, s)
    if n > 0:
        c, s = c / n, s / n
    theta = np.arctan2(s, c) % (2.0 * np.pi)
    half_frac = theta / (2.0 * np.pi) / 2.0          # in [0, 0.5)
    zero = half_frac if parity_prob >= 0.5 else half_frac + 0.5
    return {"zero_offset": float(zero % 1.0),
            "half_offset": float((zero + 0.5) % 1.0),
            "symmetry_offset": float(half_frac),
            "parity_prob": float(parity_prob)}


# ----------------------------------------------------------------------------
# 3. Model
# ----------------------------------------------------------------------------

def _torch():
    import torch
    return torch


class FluxCNN:
    """Factory for the 1-D CNN (kept out of module scope so torch stays lazy)."""

    @staticmethod
    def build(in_channels=N_CHANNELS, hidden=128):
        torch = _torch()
        nn = torch.nn

        class ConvBlock(nn.Module):
            def __init__(self, c_in, c_out, k=7, p=0.10):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(c_in, c_out, k, padding=k // 2),
                    nn.BatchNorm1d(c_out), nn.GELU(),
                    nn.Conv1d(c_out, c_out, k, padding=k // 2),
                    nn.BatchNorm1d(c_out), nn.GELU(),
                    nn.MaxPool1d(2), nn.Dropout(p))

            def forward(self, x):
                return self.net(x)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    ConvBlock(in_channels, 32, 7, 0.05),
                    ConvBlock(32, 64, 5, 0.08),
                    ConvBlock(64, 64, 5, 0.10),
                    ConvBlock(64, hidden, 3, 0.12))
                self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
                self.trunk = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                           nn.Dropout(0.15))
                self.head_angle = nn.Linear(hidden, 2)
                self.head_parity = nn.Linear(hidden, 1)

            def forward(self, x):
                h = self.trunk(self.pool(self.features(x)))
                v = self.head_angle(h)
                # unit-normalise so the head parametrises an angle, not a free
                # 2-vector; the previous model left (cos, sin) unconstrained and
                # additionally z-scored them as if they were ordinary targets.
                v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-8)
                return v, self.head_parity(h).squeeze(1)

        return Net()


def load_model(path):
    """Rebuild one checkpoint into an eval-mode model, with its training meta.

    ``weights_only=False`` is load-bearing: the checkpoint carries a plain-dict
    ``meta`` alongside the tensors, and torch >= 2.6 defaults the safe unpickler
    on.  The checkpoints must therefore come from a trusted location.

    ``meta`` records ``in_channels``, ``hidden``, ``n_points``, the device
    ``centre`` parameters, and the ``acq`` window the ensemble was trained for --
    the last of which is what lets a caller refuse a curve from outside the
    band the model knows.
    """
    torch = _torch()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = FluxCNN.build(in_channels=ckpt["meta"].get("in_channels", N_CHANNELS),
                          hidden=ckpt["meta"].get("hidden", 128))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["meta"]


def predict(models, channels) -> Dict[str, float]:
    """Run one curve through an ensemble; returns decoded offsets + confidence."""
    torch = _torch()
    if not isinstance(models, (list, tuple)):
        models = [models]
    x = torch.tensor(np.asarray(channels, np.float32)[None, ...])
    vs, ps = [], []
    with torch.no_grad():
        for m in models:
            v, logit = m(x)
            vs.append(v[0].numpy())
            ps.append(float(torch.sigmoid(logit)[0]))
    v = np.mean(vs, axis=0)
    out = decode_labels(v, float(np.mean(ps)))
    out["parity_spread"] = float(np.std(ps))
    out["angle_spread"] = float(np.std([np.arctan2(a[1], a[0]) for a in vs]))
    return out
