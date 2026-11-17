"""Convert Metrica Sports' public sample games (github.com/metrica-sports/sample-data) to npz.

These are three anonymised matches with corrected tracking and hand-validated events. They are
OTHER games than the competition match, so RULES.md allows them for training and validation.

Frame labels are derived mechanically, mirroring how CHALLENGE.md says the competition ground
truth is built from corrected tracking:

  * D  where the corrected ball position is missing for a second or more (the sample data
       blanks the ball when dead; shorter blanks are tracking drop-outs);
  * otherwise the team of the most recent on-ball event (pass, carry, shot, recovery, set piece,
    ball lost, ball out). Challenges, fouls and cards do not move possession on their own.

    python solution/tools/metrica_to_arrays.py
"""
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DIR = os.path.join(REPO, "external", "metrica")
PITCH = (105.0, 68.0)   # the sample data is normalised; the README states 105 x 68
MIN_DEAD_FRAMES = 25
ON_BALL = ("PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST", "BALL OUT")


def labels_from_events(n, ball, events):
    """events: iterable of (start_frame_index, team 0/1, type). Returns H/A/D codes 0/1/2."""
    owner = np.full(n, -1, dtype=np.int8)
    marks = sorted((f, t) for f, t, kind in events if kind in ON_BALL and 0 <= f < n)
    for i, (frame, team) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else n
        owner[frame:stop] = team
    first = marks[0][1] if marks else 0
    owner[owner < 0] = first
    # The sample tracking also blanks the ball for a few frames at a time during open play.
    # Those are tracking drop-outs, not stoppages: only a blank run of a second or more is dead.
    missing = np.isnan(ball[:, 0]).astype(np.int8)
    edges = np.diff(np.concatenate(([0], missing, [0])))
    dead = np.zeros(n, dtype=bool)
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a >= MIN_DEAD_FRAMES:
            dead[a:b] = True
    return np.where(dead, 2, owner).astype(np.int8)


def halves_from_period(period):
    out = []
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


def convert_csv_game(game):
    frames = {}
    for side in ("Home", "Away"):
        path = os.path.join(DIR, "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(game, side))
        frames[side] = pd.read_csv(path, skiprows=3, header=None).values.astype(np.float64)
    home, away = frames["Home"], frames["Away"]
    period = home[:, 0].astype(int)
    hp = home[:, 3:-2].reshape(len(home), -1, 2)
    ap = away[:, 3:-2].reshape(len(away), -1, 2)
    ball = home[:, -2:].astype(np.float32)
    xy = np.concatenate([hp, ap], axis=1).astype(np.float32)
    team = np.array([0] * hp.shape[1] + [1] * ap.shape[1], dtype=np.int8)

    ev = pd.read_csv(os.path.join(DIR, "Sample_Game_{}_RawEventsData.csv".format(game)))
    events = [(int(r["Start Frame"]) - 1, 0 if r["Team"] == "Home" else 1, r["Type"])
              for _, r in ev.iterrows()]
    label = labels_from_events(len(ball), ball, events)
    return ball, xy, team, label, halves_from_period(period)


def convert_epts_game(game=3):
    root = ET.parse(os.path.join(DIR, "Sample_Game_{}_metadata.xml".format(game))).getroot()
    teams = [t.get("id") for t in root.iter("Team")]
    players = [(p.get("id"), 0 if p.get("teamId") == teams[0] else 1) for p in root.iter("Player")]
    slot = {pid: i for i, (pid, _) in enumerate(players)}
    team = np.array([t for _, t in players], dtype=np.int8)
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    # The player set changes with substitutions: one format specification per frame range.
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        order = []
        for ref in spec.iter("PlayerChannelRef"):
            pid = channel_player[ref.get("playerChannelId")]
            if not order or order[-1] != pid:
                order.append(pid)
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")),
                      [slot[pid] for pid in order]))

    with open(os.path.join(DIR, "Sample_Game_{}_tracking.txt".format(game))) as handle:
        lines = handle.read().splitlines()
    n = len(lines)
    xy = np.full((n, len(team), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    index_of = {}
    for i, line in enumerate(lines):
        head, body, tail = line.split(":")
        frame = int(head)
        index_of[frame] = i
        slots = next(s for a, b, s in specs if a <= frame <= b)
        groups = body.split(";")
        for s, group in zip(slots, groups):
            x, y = group.split(",")
            xy[i, s] = (float(x), float(y))
        bx, by = tail.split(",")[:2]
        ball[i] = (float(bx), float(by))

    with open(os.path.join(DIR, "Sample_Game_{}_events.json".format(game))) as handle:
        data = json.load(handle)["data"]
    events = []
    for e in data:
        frame = index_of.get(int(e["start"]["frame"]))
        if frame is None or e.get("team") is None:
            continue
        events.append((frame, 0 if e["team"]["id"] == teams[0] else 1, e["type"]["name"]))
    label = labels_from_events(len(ball), ball, events)
    halves = []
    for name in ("first_half", "second_half"):
        a, b = int(float(params[name + "_start"])), int(float(params[name + "_end"]))
        ia = min(i for f, i in index_of.items() if f >= a)
        ib = max(i for f, i in index_of.items() if f <= b) + 1
        halves.append((ia, ib))
    return ball, xy, team, label, halves


if __name__ == '__main__':
    for game in (1, 2, 3):
        ball, xy, team, label, halves = convert_csv_game(game) if game < 3 else convert_epts_game()
        ball3 = np.concatenate([ball, np.zeros((len(ball), 1), np.float32)], axis=1)
        out = os.path.join(DIR, "metrica{}.npz".format(game))
        np.savez_compressed(out, ball=ball3, xy=xy, team=team, label=label,
                            halves=np.array(halves), pitch=np.array(PITCH))
        print(out, ball.shape, xy.shape, halves, np.bincount(label) / len(label))
