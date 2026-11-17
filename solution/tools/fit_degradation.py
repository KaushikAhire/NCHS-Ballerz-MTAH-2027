"""Choose the ball-noise parameters of mtah.simulate so degraded external data matches the ATD.

Only label-free statistics of the competition ball track are used as targets: quantiles of the
frame-to-frame velocity change, of the high-frequency position residual, and of the distance from
the ball to the nearest player. No ground truth of any kind is read.

    python solution/tools/fit_degradation.py
"""
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from mtah import external, features, io_match, simulate  # noqa: E402

REFERENCE = ["metrica/metrica3", "dfl/J03WMX", "dfl/J03WOH"]


def statistics(match):
    f = features.build(match)
    dv, _ = features.ball_kinematics(f)
    use = f.graded & f.visible
    dv = dv[use]
    dv = dv[~np.isnan(dv)]
    ball = f.ball
    residual = np.hypot(*(ball[1:-1] - 0.5 * (ball[:-2] + ball[2:])).T)
    residual = residual[use[1:-1] & ~np.isnan(residual)]
    nearest = np.minimum(f.near_dist[use, 0], f.near_dist[use, 1])
    return np.concatenate([np.percentile(dv, [50, 90, 99]), np.percentile(residual, [50, 90]),
                           np.percentile(nearest, [25, 50, 75])])


if __name__ == '__main__':
    target = statistics(io_match.load_match())
    print("ATD target  dv50/90/99, resid50/90, nearest25/50/75:", np.round(target, 3))
    clean = [external.load(key) for key in REFERENCE]
    grid = itertools.product((2.0, 3.0, 4.0, 5.0, 6.0), (0.35, 0.55, 0.75), (0.5, 1.0, 2.0),
                             (0.003, 0.006, 0.01))
    best = None
    for smooth, drift, tau, jitter in grid:
        noise = {"smooth_frames": smooth, "drift_m": drift, "drift_tau_s": tau,
                 "jitter_m": jitter}
        stats = np.mean([statistics(simulate.degrade(m, seed=1, visible_mean_s=None,
                                                     ball_noise=noise)) for m in clean], axis=0)
        loss = float(np.sum(np.log(stats / target) ** 2))
        if best is None or loss < best[0]:
            best = (loss, noise, stats)
            print("loss {:.3f} {} -> {}".format(loss, noise, np.round(stats, 3)), flush=True)
    print("\nchosen:", best[1])
