"""Inference side of the fluxonium offset-inverse CNN.

Given one flux period of the resonator frequency f_r(flux), the network recovers
where zero flux sits inside that period.  Training lives in the separate
Fluxonium-offset-inverse-model repository; this module only rebuilds the network
and runs a saved checkpoint, so that a protocol operation can use it.

The architecture is deliberately duplicated here rather than pickled into the
checkpoint: the checkpoint stores only ``model_state_dict``, ``config``,
``target_mean``, ``target_std`` and ``metric_names``, so the class definitions
must match the ones used at training time.  Keep them in sync with cell 1 of
"ML for resonator spectrum vs flux.ipynb".

``torch`` is imported at module level because the layer classes need it at
definition time.  Import this module lazily (inside a function) from anything
that must remain importable without torch installed.
"""

import logging
import math

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 5, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
            nn.Conv1d(c_out, c_out, kernel_size=k, padding=k // 2),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(p),
        )

    def forward(self, x):
        return self.net(x)


class FluxoniumInverseCNN(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, k=7, p=0.05),
            ConvBlock(32, 64, k=5, p=0.08),
            ConvBlock(64, 128, k=5, p=0.10),
            ConvBlock(128, hidden_dim, k=3, p=0.12),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


def preprocess_curve_for_model(curve_1d: np.ndarray) -> np.ndarray:
    """Build the (1, 3, N) input tensor the network was trained on.

    Channel 0 is the **raw curve in GHz**, channel 1 its per-curve z-score, and
    channel 2 the gradient of channel 1.  Channel 0 being absolute is what lets
    the network tell zero flux from half flux -- both are extrema of an even
    function, so only the absolute frequency breaks the symmetry.  Feeding a
    curve in the wrong units therefore does not fail loudly, it silently swaps
    the two answers.
    """
    curve = np.asarray(curve_1d, dtype=np.float32)
    c_mean = curve.mean(keepdims=True)
    c_std = curve.std(keepdims=True)
    c_std = np.where(c_std < 1e-8, 1.0, c_std)
    x_norm = (curve - c_mean) / c_std
    dx = np.gradient(x_norm).astype(np.float32)
    return np.stack([curve, x_norm, dx], axis=0)[None, ...]


def load_checkpoint(checkpoint_path):
    """Load a trained checkpoint, preferring torch's safe unpickler.

    torch >= 2.6 defaults ``weights_only=True``.  The saved dict holds plain
    primitives alongside the tensors, so the safe path normally works; the
    fallback exists for checkpoints written by older torch.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        logger.info(
            f"Safe checkpoint load failed ({exc}); retrying with weights_only=False"
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    missing = [k for k in ("model_state_dict", "config", "metric_names") if k not in ckpt]
    if missing:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing {missing}; it does not look "
            f"like a fluxonium-inverse-CNN checkpoint."
        )
    return ckpt


def predict_from_checkpoint(curve_ghz: np.ndarray, ckpt) -> dict:
    """Run an already-loaded checkpoint on one flux period of f_r.

    `curve_ghz` must be one period, uniformly spaced in flux, ascending, in GHz.
    Returns the de-normalized regression outputs keyed by the checkpoint's own
    ``metric_names``, plus ``offset_fraction_recovered`` in [0, 1) when the model
    was trained with the sin/cos offset encoding.
    """
    metric_names = list(ckpt["metric_names"])
    cfg = ckpt["config"]
    target_mean = ckpt.get("target_mean")
    target_std = ckpt.get("target_std")

    x = preprocess_curve_for_model(curve_ghz)
    x_t = torch.tensor(x, dtype=torch.float32)

    model = FluxoniumInverseCNN(
        in_channels=x.shape[1],
        out_dim=len(metric_names),
        hidden_dim=cfg["hidden_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    # Mandatory: the network has BatchNorm1d and we infer with a batch of one.
    model.eval()

    with torch.no_grad():
        pred = model(x_t).cpu().numpy()[0]

    if target_mean is not None and target_std is not None:
        pred = pred * np.asarray(target_std, dtype=np.float32) + np.asarray(
            target_mean, dtype=np.float32
        )

    result = {name: float(val) for name, val in zip(metric_names, pred)}

    if "offset_cos" in result and "offset_sin" in result:
        angle = math.atan2(result["offset_sin"], result["offset_cos"])
        if angle < 0:
            angle += 2 * math.pi
        result["offset_fraction_recovered"] = angle / (2 * math.pi)

    return result


def predict_single_curve(curve_ghz: np.ndarray, checkpoint_path) -> dict:
    """Convenience wrapper: load the checkpoint and predict in one call."""
    return predict_from_checkpoint(curve_ghz, load_checkpoint(checkpoint_path))


def offset_fraction_of(prediction: dict) -> float:
    """Pull the offset out of a prediction dict, whichever encoding was used.

    The sin/cos-trained model reports ``offset_fraction_recovered``; a model
    trained without that encoding reports ``offset_fraction`` directly.
    """
    for key in ("offset_fraction_recovered", "offset_fraction"):
        if key in prediction:
            return float(prediction[key])
    raise KeyError(
        f"No offset in prediction; got keys {sorted(prediction)}. The checkpoint "
        f"was trained without an offset target."
    )
