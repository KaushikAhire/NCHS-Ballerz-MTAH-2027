"""Leave-one-match-out evidence on the external matches.

For every external match the three evidence models are refitted WITHOUT that match and then
applied to a degraded copy of it. The decoder's hyper-parameters are tuned on these out-of-fold
predictions, so the reported external scores are never measured on data a model was fitted on.

The result is cached in solution/cache/ (several minutes to build).

    python solution/tools/oof.py [all|metrica]
"""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from mtah import external, gbm, pipeline, simulate  # noqa: E402
from mtah.io_match import CACHE_DIR  # noqa: E402

ALL = external.METRICA + external.DFL
TRAIN_SEEDS = (11, 12)
TRAIN_LEVEL = 45.0
EVAL_SEED = 1
EVAL_LEVELS = (45.0, 120.0)
THREADS = 7


def training_views(keys=ALL, seeds=TRAIN_SEEDS, level=TRAIN_LEVEL):
    """{key: [table per seed]}; a table holds the three feature views, labels and row masks."""
    out, swaps = {}, None
    for key in keys:
        clean = external.load(key)
        out[key] = []
        for seed in seeds:
            match = simulate.degrade(clean, seed=seed, visible_mean_s=level)
            v = gbm.views(match)
            f = v.frames
            alive_rows = f.graded & (match.label < 2)
            out[key].append({"players": v.players, "team": v.team, "shape": v.shape,
                             "label": match.label, "graded": f.graded,
                             "tracked": alive_rows & f.visible, "alive": alive_rows})
            swaps = (v.team_swap, v.shape_swap)
    return out, swaps


def fit(views, swaps, alive_keys, team_keys, threads=1):
    alive = gbm.train_alive([(t["players"], t["label"], t["graded"])
                             for k in alive_keys for t in views[k]], threads=threads)
    team = gbm.train_symmetric([(t["team"], t["label"], t["tracked"])
                                for k in team_keys for t in views[k]], swaps[0], "team",
                               threads=threads)
    shape = gbm.train_symmetric([(t["shape"], t["label"], t["alive"])
                                 for k in team_keys for t in views[k]], swaps[1], "shape",
                                threads=threads, step=5)
    return {"alive": alive, "team": team, "shape": shape}


def team_keys_for(hold, scheme):
    others = [k for k in ALL if k != hold]
    if scheme == "metrica":
        return [k for k in others if k in external.METRICA]
    return others


def build(scheme="all", verbose=True):
    views, swaps = training_views()
    out = {}
    for hold in ALL:
        models = fit(views, swaps, [k for k in ALL if k != hold], team_keys_for(hold, scheme),
                     threads=THREADS)
        clean = external.load(hold)
        for level in EVAL_LEVELS:
            match = simulate.degrade(clean, seed=EVAL_SEED, visible_mean_s=level)
            v = gbm.views(match)
            f = v.frames
            alive, team, shape = gbm.predict(models, v)
            ev = pipeline.evidence_from_logits(f, list(match.halves), alive, team, shape,
                                               pipeline.DEFAULT_PARAMS)
            # keep only what decode() and the scorer need
            slim = {k: ev[k] for k in ("blocks", "players_logit", "category", "lam", "team_logit",
                                       "shape_logit")}
            out[(hold, level)] = (match.label, list(match.halves), slim)
            if verbose:
                g = f.graded
                live = g & (match.label < 2)
                print("oof {:<18} level {:>5}  alive acc {:.3f} | team acc tracked {:.3f}, "
                      "untracked {:.3f} ({:.0%} of live frames untracked)".format(
                          hold, level, np.mean((alive[g] > 0) == (match.label[g] < 2)),
                          np.mean((team[live & f.visible] > 0) == (match.label[live & f.visible] == 0)),
                          np.mean((shape[live & ~f.visible] > 0) == (match.label[live & ~f.visible] == 0)),
                          np.mean(~f.visible[live])), flush=True)
    return out


def cache_path(scheme):
    return os.path.join(CACHE_DIR, "oof_{}.pkl".format(scheme))


def load(scheme="all", rebuild=False):
    path = cache_path(scheme)
    if os.path.isfile(path) and not rebuild:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    out = build(scheme)
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    with open(path, "wb") as handle:
        pickle.dump(out, handle)
    return out


if __name__ == '__main__':
    load(sys.argv[1] if len(sys.argv) > 1 else "all", rebuild=True)
