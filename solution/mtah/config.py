"""The numbers the submission was produced with.

DECODER was chosen by tools/tune.py on external matches (other games), maximising the official
spellF1 on leave-one-match-out predictions. PRIORS are fixed a priori - there is no external
data with Smart Tagging to tune them on, so they are deliberately few and conservative.
Neither was tuned on the competition match or on its public five minutes.
"""
from .priors import DEFAULT_WEIGHTS

DECODER = {
    "alive_temperature": 0.06,
    "ball_weight": 1.0,
    "cost_stop": 6.0,
    "cost_restart": 6.0,
    "team_temperature": 0.25,
    "shape_temperature": 0.1,
    "cost_turnover": 1.5,
    "min_dead": 100,
    "min_spell": 8,
}

PRIORS = dict(DEFAULT_WEIGHTS)
