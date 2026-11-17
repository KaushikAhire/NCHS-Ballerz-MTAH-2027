"""Fit the three evidence models on the external matches and write them to solution/models/.

  alive model : P(ball in play | players view)                   all seven external matches
  team model  : P(home has the ball | ball view + team shapes)   see TEAM_SCHEME below
  shape model : P(home has the ball | team shapes only)          see TEAM_SCHEME below

The external matches are degraded to look like the competition's automatic tracking before the
features are computed (mtah/simulate.py). No frame of the competition match is read here.
Single-threaded and seeded, so re-running it reproduces the committed model files.

    python solution/tools/train.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import oof  # noqa: E402
from mtah import external, gbm, simulate  # noqa: E402

TEAM_SCHEME = "all"      # "all": every external match; "metrica": the Metrica sample games only


if __name__ == '__main__':
    views, swaps = oof.training_views()
    team_keys = external.METRICA if TEAM_SCHEME == "metrica" else oof.ALL
    models = oof.fit(views, swaps, oof.ALL, team_keys, threads=1)
    gbm.save(models, {
        "alive_matches": oof.ALL, "team_matches": team_keys,
        "train_seeds": list(oof.TRAIN_SEEDS), "train_visible_mean_s": oof.TRAIN_LEVEL,
        "ball_noise": simulate.BALL_NOISE, "lightgbm": gbm.LGB_PARAMS,
        "rounds": gbm.ROUNDS,
    })
    print("wrote models to", gbm.MODEL_DIR)
