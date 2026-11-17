"""Read the provided match assets into plain numpy arrays.

Everything downstream works on one `Match` object:

    ball   (N, 2) float32   normalised pitch coordinates, NaN where the tracker has no ball
    xy     (N, S, 2) float32 normalised player positions per track slot, NaN where absent
    speed  (N, S) float32   provider speed (m/s), NaN where absent
    team   (S,) int8        0 = home, 1 = away
    halves [(first_frame, last_frame_exclusive), ...]   the graded blocks, in video frames
    pitch  (length_m, width_m)

File names are located with glob patterns rather than spelled out, so the bundle works whether it
kept the names in DATA.md or the names the download arrived with.
"""
import glob
import io
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_DATA_DIR = os.environ.get("MTAH_DATA_DIR") or os.path.join(REPO, "data")
CACHE_DIR = os.path.join(os.path.dirname(HERE), "cache")

N_FRAMES = 199319
FPS = 25
# Graded blocks, published in DATA.md (boundaries, not labels).
HALVES = [(11185, 90652), (110808, 190198)]
PITCH = (105.3, 68.0)


class Match(object):
    def __init__(self, ball, xy, speed, team, halves, pitch, fps=FPS, name="match"):
        self.ball, self.xy, self.speed, self.team = ball, xy, speed, team
        self.halves, self.pitch, self.fps, self.name = halves, pitch, fps, name
        self.n = len(ball)
        self.tags = []          # Smart Tagging instances, optional
        self.attack_dir = None  # {half_index: +1 if home attacks towards x=1 else -1}


def find_asset(data_dir, source, pattern):
    """The single file in data/<source>/ matching a glob pattern, or a readable error."""
    hits = sorted(glob.glob(os.path.join(data_dir, source, pattern)))
    if not hits:
        raise IOError("no file matching {} in {}".format(pattern, os.path.join(data_dir, source)))
    return hits[0]


def read_atd_meta(meta_path):
    """(team_of_slot, home_name, away_name, home_attack_dir_first_half)."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    home_id = teams[0].get("id")
    team = np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")],
                    dtype=np.int8)
    direction = None
    for parameter in teams[0].iter("ProviderParameter"):
        if parameter.findtext("Name") == "attack_direction_first_half":
            direction = 1 if parameter.findtext("Value") == "left_to_right" else -1
    return team, teams[0].findtext("Name"), teams[1].findtext("Name"), direction


def read_atd_raw(raw_path, n_slots):
    """Raw ATD text -> (frame_id, players (N, S, 3), ball (N, 3)). One C-engine pass."""
    with open(raw_path) as handle:
        text = handle.read()
    text = text.replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                        na_values=["NaN"], keep_default_na=True).values
    width = 1 + 3 * n_slots + 3
    if table.shape[1] != width:
        raise ValueError("{}: expected {} columns, found {}".format(
            raw_path, width, table.shape[1]))
    frame_id = table[:, 0].astype(np.int64)
    players = table[:, 1:1 + 3 * n_slots].reshape(len(table), n_slots, 3).astype(np.float32)
    ball = table[:, 1 + 3 * n_slots:].astype(np.float32)
    return frame_id, players, ball


def read_smart_tagging(path, fps=FPS):
    out = []
    for instance in ET.parse(path).iter("instance"):
        start_s, end_s = float(instance.findtext("start")), float(instance.findtext("end"))
        out.append({
            "code": (instance.findtext("code") or "").strip().upper(),
            "start": int(round(start_s * fps)), "end": int(round(end_s * fps)),
            "tags": {label.findtext("group"): (label.findtext("text") or "").strip()
                     for label in instance.findall("label")},
        })
    out.sort(key=lambda e: (e["start"], e["end"], e["code"]))
    return out


def load_match(data_dir=None, use_cache=True):
    """The tactical recording of BAR-ATH as a Match."""
    data_dir = data_dir or DEFAULT_DATA_DIR
    meta_path = find_asset(data_dir, "tactical", "*_atd*.xml")
    raw_path = find_asset(data_dir, "tactical", "*_atd*.txt")
    tag_path = find_asset(data_dir, "tactical", "*_smart_tagging*.xml")
    team, home, away, direction = read_atd_meta(meta_path)

    cache = os.path.join(CACHE_DIR, "tactical_atd.npz")
    stamp = "{}:{}".format(os.path.getsize(raw_path), int(os.path.getmtime(raw_path)))
    arrays = None
    if use_cache and os.path.isfile(cache):
        loaded = np.load(cache)
        if str(loaded["stamp"]) == stamp:
            arrays = (loaded["frame_id"], loaded["players"], loaded["ball"])
    if arrays is None:
        arrays = read_atd_raw(raw_path, len(team))
        if use_cache:
            if not os.path.isdir(CACHE_DIR):
                os.makedirs(CACHE_DIR)
            np.savez(cache, frame_id=arrays[0], players=arrays[1], ball=arrays[2], stamp=stamp)
    frame_id, players, ball = arrays
    if len(frame_id) != N_FRAMES or frame_id[0] != 1 or frame_id[-1] != N_FRAMES:
        raise ValueError("ATD is not one record per video frame numbered from 1")

    match = Match(ball[:, :2].copy(), players[:, :, :2].copy(), players[:, :, 2].copy(), team,
                  list(HALVES), PITCH, name="BAR-ATH")
    match.tags = read_smart_tagging(tag_path)
    match.team_names = (home, away)
    # Second half: teams swap ends.
    match.attack_dir = {0: direction, 1: -direction} if direction else None
    return match
