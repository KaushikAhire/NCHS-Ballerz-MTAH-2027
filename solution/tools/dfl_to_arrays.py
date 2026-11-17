"""Convert matches of the open DFL dataset (Bassek et al., Sci Data 2025, CC-BY 4.0) to npz.

The dataset carries, per 25 Hz frame, the ball position with BallStatus (1 = in play) and
BallPossession (1 = home, 2 = guest). Those two columns give a frame-level H/A/D ground truth for
matches that are NOT the competition match, which is what RULES.md allows for training and
validation.

    python solution/tools/dfl_to_arrays.py            # converts every match in external/dfl
"""
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DFL_DIR = os.path.join(REPO, "external", "dfl")

SET_RE = re.compile(r'<FrameSet GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]*)" PersonId="([^"]*)"')
NUM = r'"(-?[\d.]+)"'
FRAME_RE = re.compile(r'<Frame N="(\d+)" T="[^"]*" X=' + NUM + r' Y=' + NUM)
BALL_RE = re.compile(r'<Frame N="(\d+)" T="[^"]*" X=' + NUM + r' Y=' + NUM + r' Z=' + NUM +
                     r'.*BallPossession="(\d)" BallStatus="(\d)"')


def convert(position_path, info_path, out_path):
    root = ET.parse(info_path).getroot()
    roles = {team.get("TeamId"): team.get("Role") for team in root.iter("Team")}
    home = [t for t, r in roles.items() if r == "home"][0]
    env = next(root.iter("Environment"))
    pitch = (float(env.get("PitchX")), float(env.get("PitchY")))

    # First pass over the file: collect per-set rows.
    sets = []          # (section, team, person, frames array, values array)
    section = team = person = None
    numbers, values = [], []

    def flush():
        if section is not None and numbers:
            sets.append((section, team, person, np.array(numbers, dtype=np.int64),
                         np.array(values, dtype=np.float32)))

    with open(position_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                if team == "BALL":
                    m = BALL_RE.match(line)
                    numbers.append(int(m.group(1)))
                    values.append((float(m.group(2)), float(m.group(3)), float(m.group(4)),
                                   float(m.group(5)), float(m.group(6))))
                elif team != "referee":
                    m = FRAME_RE.match(line)
                    numbers.append(int(m.group(1)))
                    values.append((float(m.group(2)), float(m.group(3))))
            elif line.startswith("<FrameSet"):
                flush()
                m = SET_RE.match(line)
                section, team, person = m.group(1), m.group(2), m.group(3)
                numbers, values = [], []
        flush()

    sections = ["firstHalf", "secondHalf"]
    ball_sets = {s[0]: s for s in sets if s[1] == "BALL"}
    offsets, halves, start = {}, [], 0
    for name in sections:
        n = ball_sets[name][3]
        offsets[name] = (int(n[0]), start)
        halves.append((start, start + len(n)))
        start += len(n)
    total = start

    persons = sorted({(s[1], s[2]) for s in sets if s[1] not in ("BALL", "referee")},
                     key=lambda k: (k[0] != home, k[1]))
    slot = {key: i for i, key in enumerate(persons)}
    team_of_slot = np.array([0 if key[0] == home else 1 for key in persons], dtype=np.int8)

    xy = np.full((total, len(persons), 2), np.nan, dtype=np.float32)
    ball = np.full((total, 3), np.nan, dtype=np.float32)
    status = np.zeros(total, dtype=np.int8)
    possession = np.zeros(total, dtype=np.int8)
    for sec, tm, person, n, v in sets:
        if sec not in offsets:
            continue
        first, base = offsets[sec]
        index = n - first + base
        keep = (index >= 0) & (index < total)
        if tm == "BALL":
            ball[index[keep]] = v[keep, :3]
            possession[index[keep]] = v[keep, 3].astype(np.int8)
            status[index[keep]] = v[keep, 4].astype(np.int8)
        else:
            xy[index[keep], slot[(tm, person)]] = v[keep]

    # Centre-origin metres -> normalised 0-1, the convention of the competition's ATD.
    xy[:, :, 0] = xy[:, :, 0] / pitch[0] + 0.5
    xy[:, :, 1] = xy[:, :, 1] / pitch[1] + 0.5
    ball[:, 0] = ball[:, 0] / pitch[0] + 0.5
    ball[:, 1] = ball[:, 1] / pitch[1] + 0.5
    # Labels in the competition's alphabet: 0 = H, 1 = A, 2 = D.
    label = np.where(status == 1, np.where(possession == 1, 0, 1), 2).astype(np.int8)
    np.savez_compressed(out_path, ball=ball, xy=xy, team=team_of_slot, label=label,
                        halves=np.array(halves), pitch=np.array(pitch))
    return total, len(persons), halves


if __name__ == '__main__':
    for position_path in sorted(glob.glob(os.path.join(DFL_DIR, "DFL_04_03_positions_raw_observed_*.xml"))):
        key = position_path.split("positions_raw_observed_")[1][:-4]
        info_path = os.path.join(DFL_DIR, "DFL_02_01_matchinformation_{}.xml".format(key))
        out_path = os.path.join(DFL_DIR, key.split("DFL-MAT-")[1] + ".npz")
        if os.path.isfile(out_path) and "--force" not in sys.argv:
            print("have", out_path)
            continue
        print(key, convert(position_path, info_path, out_path))
