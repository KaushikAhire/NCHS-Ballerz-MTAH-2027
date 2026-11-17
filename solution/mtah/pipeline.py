"""From a Match to one H / A / D label per frame.

The same function labels the competition match and the external validation matches. The only
match-specific inputs are optional priors (Smart Tagging), passed in as per-frame arrays.
"""
import numpy as np

from . import gbm, model
from .model import H, A, D

DEFAULT_PARAMS = {
    # alive / dead
    "alive_temperature": 0.06,   # per-frame evidence is strongly autocorrelated; this tempers it
    "ball_weight": 1.0,          # weight of the adapted ball-state evidence against the players view
    "cost_stop": 6.0,            # alive -> dead
    "cost_restart": 6.0,         # dead -> alive
    # home / away
    "team_temperature": 0.25,    # evidence from the ball view (ball tracked)
    "shape_temperature": 0.1,    # evidence from the team shapes (ball not tracked)
    "cost_turnover": 4.0,        # H <-> A
    # minimum durations, frames
    "min_dead": 50,
    "min_spell": 8,
    # label-free adaptation of the ball-state evidence
    "em_iterations": 0,
    "em_temperature": 0.06,
    "em_alive_s": 30.0,
    "em_dead_s": 15.0,
    "em_clip": 4.0,
}


def evidence_from_logits(f, blocks, alive_logit, team_logit, shape_logit, params):
    category, _ = model.ball_categories(f)
    lam, share, counts = model.adapt_ball_evidence(alive_logit, category, f, blocks, params)
    return {"frames": f, "blocks": blocks, "players_logit": alive_logit, "category": category,
            "lam": lam, "share": share, "counts": counts, "team_logit": team_logit,
            "shape_logit": shape_logit}


def evidence(match, models, params):
    """Everything the decoder needs, as per-frame arrays. Split out so tuning can cache it."""
    v = gbm.views(match)
    alive, team, shape = gbm.predict(models, v)
    return evidence_from_logits(v.frames, list(match.halves), alive, team, shape, params)


def decode(ev, params, alive_prior=None, team_prior=None):
    """Viterbi over H / A / D plus minimum durations. Returns int8 codes for every frame."""
    alive = ev["players_logit"] + params["ball_weight"] * ev["lam"][ev["category"]]
    alive = params["alive_temperature"] * alive
    if alive_prior is not None:
        alive = alive + alive_prior
    team = (params["team_temperature"] * ev["team_logit"]
            + params["shape_temperature"] * ev["shape_logit"])
    if team_prior is not None:
        team = team + team_prior
    n = len(alive)
    score = np.empty((n, 3))
    # The team evidence only splits "alive" between the two teams: the favoured team scores
    # exactly what "alive" scores and the other is penalised. A confident possession call must
    # never make a frame look more alive than the alive / dead evidence says it is.
    score[:, H] = 0.5 * alive + np.minimum(team, 0.0)
    score[:, A] = 0.5 * alive + np.minimum(-team, 0.0)
    score[:, D] = -0.5 * alive
    cost = np.zeros((3, 3))
    cost[H, A] = cost[A, H] = params["cost_turnover"]
    cost[H, D] = cost[A, D] = params["cost_stop"]
    cost[D, H] = cost[D, A] = params["cost_restart"]
    path = model.viterbi(score, cost, ev["blocks"])
    path = model.enforce_minimum_durations(path, ev["blocks"], params["min_dead"],
                                           params["min_spell"], team)
    return path
