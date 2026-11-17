"""Choose the decoder's hyper-parameters on the external matches, with the official scorer.

The objective is the competition's own ranking metric (spellF1, from the organisers'
possession_metrics.py), averaged over the three Metrica sample games, each degraded at two
ball-loss levels, on leave-one-match-out predictions (tools/oof.py). The four DFL matches are
reported alongside as a check that the choice is not specific to one data provider. The
competition match and its public ground truth are never read.

Coordinate descent over a fixed grid: deterministic.

    python solution/tools/tune.py [all|metrica]     # which matches train the team model
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import oof  # noqa: E402
from mtah import external, pipeline, simulate  # noqa: E402

GRID = {
    "alive_temperature": (0.03, 0.06, 0.09, 0.13, 0.2, 0.3, 0.45, 0.7, 1.0),
    "cost_stop": (0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 9.0, 12.0),
    "cost_restart": (0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 9.0, 12.0),
    "team_temperature": (0.1, 0.17, 0.25, 0.4, 0.6, 1.0, 1.5),
    "shape_temperature": (0.0, 0.015, 0.03, 0.06, 0.1, 0.17, 0.25),
    "cost_turnover": (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0),
    "min_dead": (0, 50, 75, 100, 125, 150, 200),
    "min_spell": (0, 8, 12, 20, 30, 40, 50),
}
ORDER = ("cost_turnover", "team_temperature", "shape_temperature", "alive_temperature",
         "cost_stop", "cost_restart", "min_dead", "min_spell")
# In the degraded external data the ball is hidden exactly when it is dead, which would make the
# ball-state evidence unrealistically informative. It is therefore switched OFF while tuning: the
# alive / dead parameters are chosen for the harder case where only the players view speaks.
TUNING_OVERRIDES = {"ball_weight": 0.0}

_DATA = {}


def _init(scheme):
    _DATA.update(oof.load(scheme))


def _score(args):
    key, params = args
    label, halves, ev = _DATA[key]
    result = simulate.score(label, pipeline.decode(ev, params), halves)
    return (result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"],
            result["n_gt_spells"])


def evaluate(pool, keys, params):
    jobs = [((key, level), params) for key in keys for level in oof.EVAL_LEVELS]
    return pool.map(_score, jobs)


def objective(pool, params):
    return float(np.mean([r[0] for r in evaluate(pool, external.METRICA, params)]))


def report(pool, params):
    for label, keys in (("Metrica sample games", external.METRICA), ("DFL", external.DFL)):
        rows = evaluate(pool, keys, params)
        print("{:<22} spellF1 {:.4f}  macroF1 {:.4f}   possessions pred/true {}".format(
            label, np.mean([r[0] for r in rows]), np.mean([r[1] for r in rows]),
            ["{}/{}".format(r[2], r[3]) for r in rows]), flush=True)


STARTS = (
    {},
    {"alive_temperature": 0.3, "cost_stop": 9.0, "cost_turnover": 0.0, "team_temperature": 1.0,
     "min_spell": 20},
    {"alive_temperature": 0.13, "cost_stop": 3.0, "cost_restart": 3.0, "cost_turnover": 3.0,
     "team_temperature": 0.4, "min_spell": 12, "min_dead": 75},
)


def descend(pool, params):
    best = objective(pool, params)
    print("start  spellF1 {:.4f}".format(best), flush=True)
    for sweep in range(4):
        improved = False
        for name in ORDER:
            for value in GRID[name]:
                if value == params[name]:
                    continue
                trial = dict(params, **{name: value})
                score = objective(pool, trial)
                if score > best + 1e-4:
                    best, params, improved = score, trial, True
            print("sweep {} {:<18} -> {:<6} spellF1 {:.4f}".format(
                sweep, name, params[name], best), flush=True)
        if not improved:
            break
    return best, params


if __name__ == '__main__':
    scheme = sys.argv[1] if len(sys.argv) > 1 else "all"
    oof.load(scheme)        # build the cache in the parent before forking workers
    with Pool(7, initializer=_init, initargs=(scheme,)) as pool:
        results = [descend(pool, dict(pipeline.DEFAULT_PARAMS, **dict(TUNING_OVERRIDES, **start)))
                   for start in STARTS]
        best, params = max(results, key=lambda r: r[0])
        print("chosen (spellF1 {:.4f}):".format(best),
              json.dumps({k: params[k] for k in ORDER}))
        report(pool, params)
