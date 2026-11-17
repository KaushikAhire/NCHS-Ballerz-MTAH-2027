"""Make clean external tracking look like the competition's automatic tracking, and score it.

The external matches come from corrected or multi-camera tracking: the ball is present on every
live frame, sits within centimetres of the player who has it, and shows every kick as a sharp
change of velocity. The competition ATD is different in four label-free, measurable ways:

  * the ball is lost in long runs (almost never for less than five seconds);
  * the ball is unreliable or absent during stoppages;
  * the ball track is heavily low-pass filtered (the 99th percentile of frame-to-frame velocity
    change is ~3 m/s where clean data shows ~17 m/s);
  * the ball sits further from the nearest player (median 1.9 m against 0.3-1.4 m).

degrade() reproduces these so that models fitted on, and hyper-parameters chosen on, the external
matches see inputs that look like the ATD. The noise parameters are chosen by matching the
label-free statistics above (tools/fit_degradation.py). Everything is seeded and deterministic.
"""
import os
import sys

import numpy as np

from .io_match import Match, REPO

sys.path.insert(0, os.path.join(REPO, "evaluation"))
import possession_metrics as pm  # noqa: E402  (the organisers' scorer, used unchanged)

# Chosen by tools/fit_degradation.py against label-free statistics of the ATD ball track.
BALL_NOISE = {"smooth_frames": 6.0, "drift_m": 0.55, "drift_tau_s": 2.0, "jitter_m": 0.003}


def blackout_mask(n, fps, rng, visible_mean_s, gap_min_s=5.0, gap_extra_mean_s=15.0,
                  gap_max_s=60.0):
    mask = np.zeros(n, dtype=bool)
    t = int(rng.exponential(visible_mean_s) * fps)
    while t < n:
        gap = min(gap_max_s, gap_min_s + rng.exponential(gap_extra_mean_s))
        stop = t + int(gap * fps)
        mask[t:stop] = True
        t = stop + int(rng.exponential(visible_mean_s) * fps)
    return mask


def _gaussian_smooth_nan(values, sigma):
    """Gaussian smoothing of (N, 2) with NaN gaps: normalised convolution, NaN stays NaN."""
    if sigma <= 0:
        return values.copy()
    radius = int(np.ceil(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    valid = ~np.isnan(values[:, 0])
    weight = np.convolve(valid.astype(np.float64), kernel, mode="same")
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for axis in (0, 1):
        filled = np.where(valid, values[:, axis], 0.0)
        out[:, axis] = np.convolve(filled, kernel, mode="same") / np.maximum(weight, 1e-9)
    out[~valid] = np.nan
    return out


def noisy_ball(ball_m, fps, rng, smooth_frames, drift_m, drift_tau_s, jitter_m):
    """Ball positions in metres -> ATD-like: slow drift error, low-pass filter, fine jitter."""
    n = len(ball_m)
    rho = np.exp(-1.0 / (drift_tau_s * fps))
    white = rng.standard_normal((n, 2))
    drift = np.empty((n, 2))
    drift[0] = white[0]
    scale = np.sqrt(1.0 - rho * rho)
    for t in range(1, n):
        drift[t] = rho * drift[t - 1] + scale * white[t]
    out = _gaussian_smooth_nan(ball_m + drift_m * drift, smooth_frames)
    return out + jitter_m * rng.standard_normal((n, 2))


def degrade(match, seed=0, visible_mean_s=45.0, hide_dead=True, ball_noise=BALL_NOISE):
    rng = np.random.RandomState(seed)
    ball = match.ball.astype(np.float64).copy()
    if hide_dead:
        ball[match.label == 2] = np.nan
    if visible_mean_s:
        ball[blackout_mask(match.n, match.fps, rng, visible_mean_s)] = np.nan
    if ball_noise:
        size = np.array(match.pitch)
        ball = noisy_ball(ball * size, match.fps, rng, **ball_noise) / size
    out = Match(ball.astype(np.float32), match.xy, match.speed, match.team, match.halves,
                match.pitch, match.fps, match.name)
    out.label = match.label
    return out


def as_scored_arrays(label, pred, halves):
    """Ground truth and prediction in the scorer's layout: X outside the halves."""
    gap = 50
    gt_parts, pred_parts = [], []
    for start, stop in halves:
        gt_parts += [label[start:stop], np.full(gap, pm.X, dtype=np.int8)]
        pred_parts += [pred[start:stop], np.full(gap, pm.X, dtype=np.int8)]
    return np.concatenate(gt_parts).astype(np.int8), np.concatenate(pred_parts).astype(np.int8)


def score(label, pred, halves):
    gt, pr = as_scored_arrays(label, pred, halves)
    return pm.evaluate(gt, pr)
