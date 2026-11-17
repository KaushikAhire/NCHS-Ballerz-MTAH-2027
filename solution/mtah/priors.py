"""Match-specific prior evidence for the competition match, all of it produced by algorithms.

Two sources, both provided assets:

  * Smart Tagging (automatic, from the organisers). Three of its codes say something about the
    ball being dead, and are used as soft or hard evidence:
      - GOALS: the tagged interval runs from the goal to the following kick-off, so the ball is
        most likely dead inside it (soft evidence);
      - SET PIECES / Kick Off: play (re)starts inside the tagged five-second window. A half is
        dead until its kick-off;
      - SET PIECES (corner, goal kick, free kick, throw-in): the ball is dead when the tagged
        window opens, and the tagged team - when there is one - takes the restart.
  * The ball track itself: a tracked ball that crosses a boundary line and stays outside the
    pitch is out of play from the frame it crossed.

Nothing here is tied to a hand-picked frame: every number is read from the tag file or the
tracking by the code below.
"""
import numpy as np

HARD = 30.0     # evidence large enough to force a state through the decoder


def kickoff_frames(match, f):
    """{tag index: estimated kick-off frame} for every Kick Off tag.

    The tagger marks a window around the restart. Inside it, the kick-off is where the tracker
    re-acquires the ball in the centre circle after having lost it during the stoppage. Where
    that cannot be seen (ball tracked throughout, or never), the median offset of the kick-offs
    that could be seen in this match is used.
    """
    length, width = match.pitch
    fps = match.fps
    centre = np.hypot(f.ball[:, 0] - length / 2.0, f.ball[:, 1] - width / 2.0)
    tags = [(i, e) for i, e in enumerate(match.tags)
            if e["code"] == "SET PIECES" and "kick off" in e["tags"].get("Type", "").lower()]
    found = {}
    for i, e in tags:
        lo, hi = e["start"], min(match.n - 1, e["end"] + 2 * fps)
        for t in range(max(lo, fps), hi):
            reacquired = f.visible[t] and not f.visible[t - fps:t].any()
            if reacquired and centre[t] < 9.15:
                found[i] = t
                break
    offsets = [found[i] - e["start"] for i, e in tags if i in found]
    fallback = int(np.median(offsets)) if offsets else int(0.6 * 5 * fps)
    return {i: found.get(i, e["start"] + fallback) for i, e in tags}, tags


def out_of_play_episodes(match, f, min_frames=12, margin=0.5, max_speed=15.0):
    """[(start, stop)] runs where the tracked ball has left the pitch and is really out.

    Kept only if the ball was tracked crossing the line (not a detection that appears outside),
    stays out for half a second or the track ends there, gets clearly beyond the line, and is not
    moving at the speed of a ball in flight (a high ball near the far touchline projects outside
    the pitch for a few frames).
    """
    out = f.visible & (np.nan_to_num(f.out_dist, nan=-1.0) > 0.0)
    edges = np.diff(np.concatenate(([0], out.astype(np.int8), [0])))
    episodes = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        crossed = a > 0 and f.visible[a - 1]
        lost_after = b >= f.n or not f.visible[b]
        long_enough = (b - a) >= min_frames or lost_after
        far_enough = np.nanmax(f.out_dist[a:b]) >= margin
        speed = f.ball_speed[a:b]
        slow_enough = np.isnan(speed).all() or np.nanmedian(speed) < max_speed
        if crossed and long_enough and far_enough and slow_enough:
            episodes.append((int(a), int(b)))
    return episodes


def build(match, f, weights):
    """(alive_prior, team_prior, notes): per-frame additions to the decoder's evidence."""
    n, fps = match.n, match.fps
    alive = np.zeros(n)
    team = np.zeros(n)
    notes = {}

    kick, kick_tags = kickoff_frames(match, f)
    notes["kickoffs"] = sorted(kick.values())
    # 1. a half is dead until its kick-off
    for start, stop in match.halves:
        first = [t for t in kick.values() if start - 10 * fps <= t < stop]
        if first:
            alive[start:max(start, min(first))] -= HARD
    # 2. goal -> kick-off is dead
    goals = [e for e in match.tags if e["code"] == "GOALS"]
    notes["goals"] = []
    for goal in goals:
        after = [t for t in kick.values() if goal["start"] < t <= goal["end"] + 10 * fps]
        end = min(after) if after else goal["end"]
        begin = goal["start"] + int(weights["goal_delay_s"] * fps)
        alive[begin:end] -= weights["goal"]
        notes["goals"].append((begin, end))
        # the side that conceded takes the kick-off
        side = _side(goal["tags"].get("Team"), match)
        if side is not None:
            sign = -1.0 if side == 0 else 1.0     # home scored -> away kicks off
            team[end:end + int(weights["restart_team_s"] * fps)] += sign * weights["restart_team"]
    # 3. every other set piece: dead as the window opens, tagged team restarts
    for e in match.tags:
        if e["code"] != "SET PIECES" or "kick off" in e["tags"].get("Type", "").lower():
            continue
        a = e["start"]
        alive[a:a + int(weights["set_piece_s"] * fps)] -= weights["set_piece"]
        side = _side(e["tags"].get("Team"), match)
        if side is not None:
            sign = 1.0 if side == 0 else -1.0
            team[a:a + int(weights["restart_team_s"] * fps)] += sign * weights["restart_team"]
    # 4. ball tracked out of the pitch
    episodes = out_of_play_episodes(match, f)
    notes["out_of_play"] = len(episodes)
    for a, b in episodes:
        alive[a:b] -= weights["out_of_play"]
    return alive, team, notes


def _side(name, match):
    """0 for the home club, 1 for the away club, None for N/A. Names come from the ATD metadata."""
    if not name or name.strip().upper() == "N/A":
        return None
    home, away = match.team_names
    key = name.strip().lower()
    if key == home.strip().lower():
        return 0
    if key == away.strip().lower():
        return 1
    return None


DEFAULT_WEIGHTS = {
    "goal": 0.15,              # per frame from the goal tag to the following kick-off (soft:
                               # the tagger can be wrong, strong contrary evidence still wins)
    "goal_delay_s": 2.0,       # the goal tag opens on the goal; allow the ball to cross the line
    "set_piece": 0.5,          # per frame, for set_piece_s seconds from the tag start
    "set_piece_s": 1.0,
    "restart_team": 0.05,      # per frame, for restart_team_s seconds: who takes the restart
    "restart_team_s": 15.0,
    "out_of_play": 1.0,        # per frame while the ball is tracked outside the pitch
}
