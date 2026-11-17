"""Load the converted external matches (other games, with frame labels) as Match objects."""
import os

import numpy as np

from .io_match import Match, REPO

EXTERNAL = os.path.join(REPO, "external")
METRICA = ["metrica/metrica1", "metrica/metrica2", "metrica/metrica3"]
DFL = ["dfl/J03WMX", "dfl/J03WN1", "dfl/J03WOH", "dfl/J03WPY"]


def load(key):
    z = np.load(os.path.join(EXTERNAL, key + ".npz"))
    xy = z["xy"]
    match = Match(z["ball"][:, :2].copy(), xy, np.full(xy.shape[:2], np.nan, np.float32),
                  z["team"], [tuple(h) for h in z["halves"].tolist()],
                  tuple(float(v) for v in z["pitch"]), name=key)
    match.label = z["label"].copy()
    match.ball_z = z["ball"][:, 2].copy()
    return match
