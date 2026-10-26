"""Mask-series metrics used to judge processing (and later in analysis).

Reworking metrics follow the planform-change framework of Greenberg & Ganti
(2024): for masks C_t of the channel in year t,

* A_M(Δt) = mean over t of |C_t ∩ C_{t+Δt}| / |C_t| — the share of the
  channel still channel Δt years later (memory);
* A_R(Δt) = mean over t of |(C_{t} ∪ … ∪ C_{t+Δt}) \\ C_t| / |C_t| — new area
  visited within Δt years, relative to the channel (reworking).

Spurious year-to-year noise inflates A_R and deflates A_M at every lag, so
comparing them with the truth on synthetic rivers measures how much of the
processed signal is real change.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union (1 when both are empty)."""
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def false_share(pred: np.ndarray, truth: np.ndarray) -> float:
    """Share of predicted channel that is not channel in the truth."""
    n = pred.sum()
    return 0.0 if n == 0 else float((pred & ~truth).sum() / n)


def memory_reworking(
    masks: Sequence[np.ndarray | None], lags: Sequence[int] = (1, 2, 5, 10)
) -> dict[int, tuple[float, float]]:
    """{Δt: (A_M, A_R)}; ``None`` masks (no data) are skipped as starts and
    as members of the union."""
    out = {}
    T = len(masks)
    for d in lags:
        am, ar = [], []
        for t in range(T - d):
            c0, c1 = masks[t], masks[t + d]
            if c0 is None or c1 is None or c0.sum() == 0:
                continue
            n = c0.sum()
            am.append((c0 & c1).sum() / n)
            u = np.zeros_like(c0)
            for s in range(t, t + d + 1):
                if masks[s] is not None:
                    u |= masks[s]
            ar.append((u & ~c0).sum() / n)
        out[d] = (float(np.mean(am)), float(np.mean(ar))) if am else (np.nan, np.nan)
    return out
