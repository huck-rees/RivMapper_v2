"""Per-pixel hidden Markov model of annual DSWE counts.

Each pixel's yearly record of clear looks is treated as noisy evidence
about a hidden state that changes rarely:

* 3 states: ``dry``, ``margin`` (wet only at higher stage), ``channel``;
* 2 states: ``dry``, ``channel``.

**Observation.** A year's clear looks ``n_obs`` are split three ways:
strict wet c1 = ``n_wet_2`` (DSWE 1–2), loose-only wet c2 = ``n_wet_3`` −
``n_wet_2`` (DSWE 3), and dry c0 = ``n_obs`` − ``n_wet_3``. Given state s,
(c0, c1, c2) is multinomial with probabilities θ_s. A year with no clear
looks carries no evidence (likelihood 1), so the model fills it from the
neighbouring years in proportion to how much those observed.

**Change.** A yearly transition matrix A (every move allowed, including dry
straight to channel for avulsions) and initial probabilities π.

All parameters are fitted per river by expectation–maximisation
(Baum–Welch) on a sample of pixels, with no labels. Posteriors P(state)
per pixel and year then come from the forward–backward algorithm, run in
chunks of pixels to bound memory.

Reference for the approach on annual land-cover maps: Lamarche et al.
(2022), IEEE TGRS, doi:10.1109/TGRS.2021.3123738.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

STATE_NAMES = {2: ("dry", "channel"), 3: ("dry", "margin", "channel")}
_INIT_THETA = {
    2: np.array([[0.95, 0.02, 0.03], [0.10, 0.80, 0.10]]),
    3: np.array([[0.95, 0.02, 0.03], [0.55, 0.30, 0.15], [0.10, 0.80, 0.10]]),
}
_EPS = 1e-6


def observations(n_obs, n_wet_2, n_wet_3) -> np.ndarray:
    """(…, 3) int32 array of (dry, strict wet, loose-only wet) looks.

    Inputs are count arrays of any matching shape; NODATA (255) must be
    removed by the caller.
    """
    n = np.asarray(n_obs, np.int32)
    w2 = np.asarray(n_wet_2, np.int32)
    w3 = np.asarray(n_wet_3, np.int32)
    if (w2 > w3).any() or (w3 > n).any():
        raise ValueError("counts are not nested: need n_wet_2 <= n_wet_3 <= n_obs")
    return np.stack([n - w3, w2, w3 - w2], axis=-1)


@dataclass
class HMMParams:
    """Fitted model. ``theta[s]`` = (P dry look, P strict wet, P loose wet)."""

    theta: np.ndarray
    A: np.ndarray
    pi: np.ndarray
    loglik: float = float("nan")
    iterations: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def n_states(self) -> int:
        return len(self.pi)

    @property
    def names(self) -> tuple[str, ...]:
        return STATE_NAMES[self.n_states]

    def to_dict(self) -> dict:
        """Plain-Python form for the run record."""
        return {
            "states": list(self.names),
            "theta_dry_strict_loose": {
                n: [round(float(v), 4) for v in row]
                for n, row in zip(self.names, self.theta, strict=True)
            },
            "transition": {
                n: [round(float(v), 4) for v in row]
                for n, row in zip(self.names, self.A, strict=True)
            },
            "initial": [round(float(v), 4) for v in self.pi],
            "loglik_per_pixel_year": round(self.loglik, 4),
            "iterations": self.iterations,
            "notes": self.notes,
        }

    @classmethod
    def initial(cls, n_states: int) -> HMMParams:
        S = n_states
        A = np.full((S, S), 0.05 / (S - 1))
        np.fill_diagonal(A, 0.95)
        pi = np.full(S, 0.1 / (S - 1))
        pi[0] = 0.9
        return cls(theta=_INIT_THETA[S].copy(), A=A, pi=pi)


def log_emissions(obs: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """(T, P, S) log-likelihood of each year's looks under each state.

    The multinomial coefficient is the same for every state and is
    omitted (it cancels in posteriors and in EM).
    """
    logt = np.log(np.clip(theta, _EPS, 1.0)).astype(np.float32)  # (S, 3)
    return obs.astype(np.float32) @ logt.T


def forward_backward(logE: np.ndarray, A: np.ndarray, pi: np.ndarray, want_xi=False):
    """Scaled forward–backward.

    Args:
        logE: (T, P, S) log emissions.
        A: (S, S) transitions, rows = from.
        pi: (S,) initial probabilities.
        want_xi: Also return expected transition counts summed over
            years and pixels (for EM).

    Returns:
        gamma (T, P, S) float32 posteriors, loglik (total), and xi (S, S)
        or None.
    """
    T, P, S = logE.shape
    shift = logE.max(axis=2, keepdims=True)
    E = np.exp(logE - shift)  # (T, P, S), max 1 per (t, p)
    alpha = np.empty_like(E)
    c = np.empty((T, P), np.float64)
    a = pi[None, :] * E[0]
    c[0] = a.sum(1)
    alpha[0] = a / c[0][:, None]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * E[t]
        c[t] = a.sum(1)
        alpha[t] = a / c[t][:, None]
    beta = np.ones_like(E)
    xi = np.zeros((S, S)) if want_xi else None
    for t in range(T - 2, -1, -1):
        b = E[t + 1] * beta[t + 1]  # (P, S)
        if want_xi:
            xi += A * (alpha[t].T @ (b / c[t + 1][:, None]))
        beta[t] = (b @ A.T) / c[t + 1][:, None]
    gamma = alpha * beta
    gamma /= gamma.sum(2, keepdims=True)
    loglik = float(np.log(c).sum() + shift.sum())
    return gamma.astype(np.float32), loglik, xi


def fit(
    obs: np.ndarray,
    n_states: int = 3,
    max_iter: int = 30,
    tol: float = 1e-4,
    params: HMMParams | None = None,
) -> HMMParams:
    """Fit by Baum–Welch on a sample.

    Args:
        obs: (T, P, 3) looks from :func:`observations`, P sampled pixels.
        n_states: 2 or 3.
        max_iter, tol: Stop when the log-likelihood gain per pixel-year is
            below ``tol``.
        params: Starting values (default :meth:`HMMParams.initial`).

    Returns:
        Fitted parameters, states ordered by their strict-wet probability
        (dry < margin < channel) so labels cannot swap. For 3 states, if
        the margin state ends up holding < 0.5% of pixel-years, the model
        is refitted with 2 states and a note is recorded.
    """
    T, P, _ = obs.shape
    p = params or HMMParams.initial(n_states)
    theta, A, pi = p.theta.copy(), p.A.copy(), p.pi.copy()
    n_py = T * P
    prev = -np.inf
    it = 0
    for it in range(1, max_iter + 1):
        gamma, ll, xi = forward_backward(log_emissions(obs, theta), A, pi, True)
        # M-step
        w = gamma.reshape(-1, gamma.shape[2])  # (TP, S)
        o = obs.reshape(-1, 3).astype(np.float64)
        num = w.T.astype(np.float64) @ o  # (S, 3)
        theta = (num + 1e-3) / (num + 1e-3).sum(1, keepdims=True)
        A = (xi + 1e-6) / (xi + 1e-6).sum(1, keepdims=True)
        pi = gamma[0].mean(0).astype(np.float64)
        pi = (pi + 1e-6) / (pi + 1e-6).sum()
        gain = (ll - prev) / n_py
        prev = ll
        if gain < tol:
            break
    order = np.argsort(theta[:, 1])
    out = HMMParams(
        theta=theta[order],
        A=A[np.ix_(order, order)],
        pi=pi[order],
        loglik=prev / n_py,
        iterations=it,
    )
    if n_states == 3:
        gamma, _, _ = forward_backward(log_emissions(obs, out.theta), out.A, out.pi)
        margin_share = float((gamma.argmax(2) == 1).mean())
        if margin_share < 0.005:
            log.info(
                "margin state holds %.2f%% of pixel-years; refitting 2 states",
                100 * margin_share,
            )
            two = fit(obs, 2, max_iter, tol)
            two.notes.append(
                f"3-state fit degenerate (margin {margin_share:.4f}); used 2 states"
            )
            return two
    return out


def posteriors(
    obs: np.ndarray, params: HMMParams, chunk: int = 100_000
) -> tuple[np.ndarray, np.ndarray]:
    """P(channel) and P(margin) for every pixel and year.

    Args:
        obs: (T, P, 3) looks.
        params: Fitted parameters.
        chunk: Pixels per forward–backward pass (bounds memory).

    Returns:
        (p_channel, p_margin), each (T, P) float32. ``p_margin`` is zeros
        for a 2-state model.
    """
    T, P, _ = obs.shape
    p_ch = np.empty((T, P), np.float32)
    p_mg = np.zeros((T, P), np.float32)
    for s in range(0, P, chunk):
        g, _, _ = forward_backward(
            log_emissions(obs[:, s : s + chunk], params.theta), params.A, params.pi
        )
        p_ch[:, s : s + chunk] = g[..., -1]
        if params.n_states == 3:
            p_mg[:, s : s + chunk] = g[..., 1]
    return p_ch, p_mg
