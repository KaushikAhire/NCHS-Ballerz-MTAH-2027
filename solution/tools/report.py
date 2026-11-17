"""External validation table: the pipeline against the organisers' baseline, on other games.

Every number is leave-one-match-out (tools/oof.py): the models that label a match never saw it.
The baseline is the repository's own (nearest tracked player, +/- 50 frame majority vote) run on
exactly the same degraded input. Scored with the organisers' possession_metrics.py.

    python solution/tools/report.py [all|metrica]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import oof  # noqa: E402
from mtah import config, external, features, pipeline, simulate  # noqa: E402


def baseline_labels(f, window=50):
    """starter/baseline.py, on a Frames object: nearest player's team, D where no ball, smoothed."""
    codes = np.where(f.visible, np.where(f.near_dist[:, 0] <= f.near_dist[:, 1], 0, 1), -1)
    home = np.concatenate(([0], np.cumsum(codes == 0)))
    away = np.concatenate(([0], np.cumsum(codes == 1)))
    index = np.arange(len(codes))
    low, high = np.maximum(0, index - window), np.minimum(len(codes), index + window + 1)
    h, a = home[high] - home[low], away[high] - away[low]
    out = np.full(len(codes), 2, dtype=np.int8)
    seen = (h + a) > 0
    out[seen] = np.where(h[seen] >= a[seen], 0, 1)
    return out


if __name__ == '__main__':
    scheme = sys.argv[1] if len(sys.argv) > 1 else "all"
    data = oof.load(scheme)
    params = dict(pipeline.DEFAULT_PARAMS, **config.DECODER)
    params["ball_weight"] = 0.0      # see tools/tune.py: not meaningful on the degraded data
    print("{:<18} {:>6} | {:>8} {:>8} {:>10} | {:>8} {:>8} {:>10}".format(
        "match", "loss", "spellF1", "macroF1", "poss", "base sF1", "base mF1", "base poss"))
    rows = {"metrica": [], "dfl": []}
    for key in oof.ALL:
        clean = external.load(key)
        for level in oof.EVAL_LEVELS:
            label, halves, ev = data[(key, level)]
            ours = simulate.score(label, pipeline.decode(ev, params), halves)
            match = simulate.degrade(clean, seed=oof.EVAL_SEED, visible_mean_s=level)
            base = simulate.score(label, baseline_labels(features.build(match)), halves)
            rows["metrica" if key in external.METRICA else "dfl"].append(
                (ours["spell_f1_mean"], ours["macro_f1"], base["spell_f1_mean"], base["macro_f1"]))
            print("{:<18} {:>6} | {:>8.4f} {:>8.4f} {:>10} | {:>8.4f} {:>8.4f} {:>10}".format(
                key, "heavy" if level == oof.EVAL_LEVELS[0] else "light",
                ours["spell_f1_mean"], ours["macro_f1"],
                "{}/{}".format(ours["n_pred_spells"], ours["n_gt_spells"]),
                base["spell_f1_mean"], base["macro_f1"],
                "{}/{}".format(base["n_pred_spells"], base["n_gt_spells"])), flush=True)
    for name, values in rows.items():
        mean = np.mean(values, axis=0)
        print("mean {:<8} spellF1 {:.4f}  macroF1 {:.4f}   | baseline spellF1 {:.4f}  macroF1 {:.4f}"
              .format(name, *mean))
