"""The three learned evidence models, as gradient-boosted trees (LightGBM).

  alive model : P(ball in play | players view)                    every frame
  team model  : P(home has the ball | ball view + team shapes)    frames where the ball is tracked
  shape model : P(home has the ball | team shapes only)           frames where it is not

All three are trained only on OTHER games (see tools/train.py). Training is deterministic: fixed
seed, single thread, LightGBM's deterministic mode. The two possession models are symmetrised:
every training example is also presented with the teams exchanged, and predictions average the
two orientations, so neither model can prefer a side.
"""
import json
import os

import numpy as np

from . import features

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(HERE), "models")

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=300,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=0, deterministic=True, force_row_wise=True, num_threads=1)
SHAPE_PARAMS = dict(LGB_PARAMS, num_leaves=15, min_data_in_leaf=1000)
ROUNDS = {"alive": 300, "team": 400, "shape": 200}
EPS = 1e-4


class Views(object):
    """Feature tables of one match."""


def views(match):
    v = Views()
    f = features.build(match)
    v.frames = f
    v.players, v.players_names = features.players_view2(f, match)
    ball, ball_names, ball_swap = features.team_view2(f)
    shape, shape_names, shape_swap = features.shape_view(f, match)
    v.shape, v.shape_names, v.shape_swap = shape, shape_names, shape_swap
    v.team = np.hstack([ball, shape])
    v.team_names = ball_names + shape_names
    v.team_swap = np.concatenate([ball_swap, len(ball_names) + shape_swap])
    return v


def _train(params, x, y, rounds, threads):
    import lightgbm as lgb
    return lgb.train(dict(params, num_threads=threads), lgb.Dataset(x, y), num_boost_round=rounds)


def train_alive(tables, threads=1):
    """tables: list of (players view, label codes, graded mask)."""
    x = np.vstack([pv[g][::3] for pv, _, g in tables])
    y = np.concatenate([(label[g][::3] < 2) for _, label, g in tables]).astype(np.int32)
    return _train(LGB_PARAMS, x, y, ROUNDS["alive"], threads)


def train_symmetric(tables, swap, kind, threads=1, step=2):
    """tables: list of (feature view, label codes, usable mask). kind: 'team' or 'shape'."""
    xs, ys = [], []
    for view, label, use in tables:
        x = view[use][::step]
        y = (label[use][::step] == 0).astype(np.int32)
        xs += [x, x[:, swap]]
        ys += [y, 1 - y]
    params = SHAPE_PARAMS if kind == "shape" else LGB_PARAMS
    return _train(params, np.vstack(xs), np.concatenate(ys), ROUNDS[kind], threads)


def to_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def predict_alive_logit(booster, players):
    return to_logit(booster.predict(players, num_threads=1))


def predict_symmetric_logit(booster, view, swap, rows):
    """Exactly antisymmetric in the two teams; zero outside `rows`."""
    out = np.zeros(len(view))
    if rows.any():
        x = view[rows]
        p = 0.5 * (booster.predict(x, num_threads=1)
                   + 1.0 - booster.predict(x[:, swap], num_threads=1))
        out[rows] = to_logit(p)
    return out


def predict(models, v):
    """(alive logit, team logit on tracked-ball frames, shape logit on the other frames)."""
    f = v.frames
    alive = predict_alive_logit(models["alive"], v.players)
    team = predict_symmetric_logit(models["team"], v.team, v.team_swap, f.visible)
    in_play_block = f.graded & ~f.visible
    shape = predict_symmetric_logit(models["shape"], v.shape, v.shape_swap, in_play_block)
    return alive, team, shape


def save(models, meta):
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    for name, booster in models.items():
        booster.save_model(os.path.join(MODEL_DIR, name + ".txt"))
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)


def load():
    import lightgbm as lgb
    return {name: lgb.Booster(model_file=os.path.join(MODEL_DIR, name + ".txt"))
            for name in ("alive", "team", "shape")}
