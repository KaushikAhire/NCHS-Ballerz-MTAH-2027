"""Per-frame features. Everything here is label-free and identical for every match.

Three independent "views" of a frame are built:

  * the PLAYERS view (players_view2) - how fast the players are moving, now and a few seconds
    either side. It never looks at the ball, so it exists on every frame, including the 41% where
    the tracker has lost the ball. It drives the alive / dead decision.
  * the BALL view (team_view2) - the distance from the ball to the nearest player of each team,
    over a window of time, plus how the ball itself moves. Only defined where the ball is
    tracked. Together with the shape view it drives the home / away decision on those frames.
  * the SHAPE view (shape_view) - how wide, how deep and how fast each team is. The team in
    possession spreads out and the defending team narrows, so this says who has the ball even
    when the ball cannot be seen. It drives the home / away decision where the ball is lost.
"""
import numpy as np


def moving_mean(values, half_window):
    """NaN-aware centred moving average over +/- half_window frames (cumulative sums)."""
    valid = ~np.isnan(values)
    filled = np.where(valid, values, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(filled, dtype=np.float64)))
    ccnt = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_window)
    high = np.minimum(len(values), index + half_window + 1)
    count = ccnt[high] - ccnt[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (csum[high] - csum[low]) / np.maximum(count, 1), np.nan)


def velocity(position, fps, step=2, max_speed=13.0):
    """Central-difference velocity of (N, ..., 2) positions in metres, one-sided at track ends.

    A displacement implying more than max_speed m/s is a tracker identity switch, not a sprint,
    and is blanked rather than believed.
    """
    n = len(position)
    out = np.full(position.shape, np.nan, dtype=np.float32)
    dt = 1.0 / fps
    out[step:n - step] = (position[2 * step:] - position[:-2 * step]) / (2 * step * dt)
    forward = np.full(position.shape, np.nan, dtype=np.float32)
    forward[:n - step] = (position[step:] - position[:-step]) / (step * dt)
    backward = np.full(position.shape, np.nan, dtype=np.float32)
    backward[step:] = (position[step:] - position[:-step]) / (step * dt)
    missing = np.isnan(out[..., 0])
    out[missing] = forward[missing]
    missing = np.isnan(out[..., 0])
    out[missing] = backward[missing]
    with np.errstate(invalid="ignore"):
        speed = np.hypot(out[..., 0], out[..., 1])
        out[speed > max_speed] = np.nan
    return out


def shift(values, lag):
    """values[t + lag], clamped at the ends."""
    index = np.clip(np.arange(len(values)) + lag, 0, len(values) - 1)
    return values[index]


def quantile_normalise(values, mask):
    """Rank of each value among the frames in `mask`, scaled to 0-1.

    Speeds estimated from different tracking systems differ by noise and smoothing, but the
    ordering of frames within one match is comparable. Ranking makes a model fitted on other
    matches portable to the automatic tracking of the competition match.
    """
    reference = np.sort(values[mask & ~np.isnan(values)])
    out = np.searchsorted(reference, values, side="left") / float(max(1, len(reference)))
    out[np.isnan(values)] = 0.5
    return out.astype(np.float32)


class Frames(object):
    """Derived per-frame arrays for one Match (all lengths N)."""


def build(match):
    length, width = match.pitch
    fps = match.fps
    n = match.n
    f = Frames()
    f.n = n
    f.fps = fps
    f.pitch = match.pitch
    f.graded = np.zeros(n, dtype=bool)
    for start, stop in match.halves:
        f.graded[start:stop] = True

    pos = np.stack([match.xy[:, :, 0] * length, match.xy[:, :, 1] * width], axis=2)
    f.pos = pos
    vel = velocity(pos, fps)
    with np.errstate(invalid="ignore"):
        speed = np.hypot(vel[..., 0], vel[..., 1])
    f.vel, f.speed = vel, speed

    # ---- players view -------------------------------------------------------------------------
    present = ~np.isnan(speed)
    count = present.sum(axis=1)
    f.n_players = (~np.isnan(pos[..., 0])).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        enough = count > 3
        mean_speed = np.where(enough, np.nansum(speed, axis=1) / np.maximum(count, 1), np.nan)
    f.mean_speed = mean_speed

    # ---- ball view ----------------------------------------------------------------------------
    ball = np.stack([match.ball[:, 0] * length, match.ball[:, 1] * width], axis=1)
    f.ball = ball
    f.visible = ~np.isnan(ball[:, 0])
    bvel = velocity(ball, fps, max_speed=60.0)
    f.ball_vel = bvel
    with np.errstate(invalid="ignore"):
        f.ball_speed = np.hypot(bvel[:, 0], bvel[:, 1])
        # > 0 outside the pitch, NaN if no ball
        f.out_dist = np.maximum.reduce([-ball[:, 0], ball[:, 0] - length, -ball[:, 1],
                                        ball[:, 1] - width])
        dist = np.hypot(pos[..., 0] - ball[:, None, 0], pos[..., 1] - ball[:, None, 1])
        rel = np.hypot(vel[..., 0] - bvel[:, None, 0], vel[..., 1] - bvel[:, None, 1])
    dist[np.isnan(dist)] = np.inf
    f.near_dist = np.full((n, 2), np.inf, dtype=np.float32)
    f.near_rel = np.full((n, 2), np.nan, dtype=np.float32)
    rows = np.arange(n)
    for side in (0, 1):
        sub = dist[:, match.team == side]
        arg = np.argmin(sub, axis=1)
        f.near_dist[:, side] = sub[rows, arg]
        f.near_rel[:, side] = rel[:, match.team == side][rows, arg]
    f.near_dist[~f.visible] = np.inf
    return f


# ================================================================================================
# The three views, consumed by gradient-boosted trees.
# ================================================================================================

def _time_since(event, cap):
    """Frames since the most recent True in `event` (cap where none yet)."""
    n = len(event)
    index = np.where(event, np.arange(n), -10 ** 9)
    last = np.maximum.accumulate(index)
    return np.minimum(np.arange(n) - last, cap).astype(np.float32)


def _time_until(event, cap):
    return _time_since(event[::-1], cap)[::-1]


def _rolling(values, window, kind, future=False):
    """Rolling min/max over the past (or future) `window` frames, inclusive. NaN-tolerant."""
    fill = np.inf if kind == "min" else -np.inf
    filled = np.where(np.isnan(values), fill, values).astype(np.float64)
    n = len(filled)
    pad = np.full(window - 1, fill)
    padded = np.concatenate([filled, pad]) if future else np.concatenate([pad, filled])
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    out = view.min(axis=1) if kind == "min" else view.max(axis=1)
    out = out[:n].copy()
    out[np.isinf(out)] = np.nan
    return out.astype(np.float32)


TEAM_LAGS = (-50, -25, -12, -6, -3, 0, 3, 6, 12, 25, 50)
BALL_LAGS = (-12, -6, 0, 6, 12)
TOUCH_DV = 1.5        # m/s change of (low-pass filtered) ball velocity that counts as a touch
TOUCH_RADIUS = 2.5    # m, a player must be this close for the touch to be theirs
CLOSE_RADIUS = 1.5


def ball_kinematics(f):
    """Velocity-change magnitude of the ball and the detected touches (team 0/1, else -1)."""
    n, fps = f.n, f.fps
    ball = f.ball
    forward = np.full((n, 2), np.nan, dtype=np.float32)
    backward = np.full((n, 2), np.nan, dtype=np.float32)
    forward[:-2] = (ball[2:] - ball[:-2]) * (fps / 2.0)
    backward[2:] = (ball[2:] - ball[:-2]) * (fps / 2.0)
    with np.errstate(invalid="ignore"):
        dv = np.hypot(forward[:, 0] - backward[:, 0], forward[:, 1] - backward[:, 1])
    filled = np.where(np.isnan(dv), 0.0, dv)
    local_max = np.maximum(_rolling(filled, 3, "max", False), _rolling(filled, 3, "max", True))
    nearest = np.minimum(f.near_dist[:, 0], f.near_dist[:, 1])
    touch = (filled >= local_max) & (filled > TOUCH_DV) & (nearest < TOUCH_RADIUS)
    team = np.where(f.near_dist[:, 0] <= f.near_dist[:, 1], 0, 1)
    return dv.astype(np.float32), np.where(touch, team, -1).astype(np.int8)


def team_view2(f):
    """(features (N, k), names, swap_index). Rows where the ball is not tracked are all-NaN.

    Features come in (home, away) pairs plus a block shared by both teams. `swap_index` permutes
    the columns into the same description with the teams exchanged, which is how the model is
    made exactly symmetric at training and at prediction time.
    """
    fps = f.fps
    cap = 10 * fps
    d = np.where(np.isinf(f.near_dist), np.nan, f.near_dist).astype(np.float32)
    dv, touch = ball_kinematics(f)
    f.ball_dv, f.touch = dv, touch
    pairs, shared, names_p, names_s = [], [], [], []

    for lag in TEAM_LAGS:
        pairs.append((shift(d[:, 0], lag), shift(d[:, 1], lag)))
        names_p.append("dist@{:+d}".format(lag))
    pairs.append((f.near_rel[:, 0], f.near_rel[:, 1]))
    names_p.append("relspeed")
    close = [np.nan_to_num(d[:, side], nan=99.0) < CLOSE_RADIUS for side in (0, 1)]
    for name, events in (("touch", [touch == 0, touch == 1]), ("close", close)):
        pairs.append((_time_since(events[0], cap), _time_since(events[1], cap)))
        names_p.append("since_" + name)
        pairs.append((_time_until(events[0], cap), _time_until(events[1], cap)))
        names_p.append("until_" + name)
    for future in (False, True):
        pairs.append((_rolling(d[:, 0], fps, "min", future), _rolling(d[:, 1], fps, "min", future)))
        names_p.append("mindist_" + ("next1s" if future else "last1s"))

    speed = f.ball_speed.astype(np.float32)
    for lag in BALL_LAGS:
        shared.append(shift(speed, lag))
        names_s.append("ballspeed@{:+d}".format(lag))
    shared.append(dv)
    names_s.append("dv")
    shared.append(_rolling(dv, fps // 2, "max", False))
    names_s.append("dv_last0.5s")
    shared.append(_rolling(dv, fps // 2, "max", True))
    names_s.append("dv_next0.5s")
    shared.append(moving_mean(f.visible.astype(np.float32), fps).astype(np.float32))
    names_s.append("visible_frac_2s")

    columns, names, swap = [], [], []
    for (home, away), name in zip(pairs, names_p):
        swap += [len(columns) + 1, len(columns)]
        columns += [home, away]
        names += [name + "_H", name + "_A"]
    for values, name in zip(shared, names_s):
        swap.append(len(columns))
        columns.append(values)
        names.append(name)
    out = np.stack(columns, axis=1).astype(np.float32)
    out[~f.visible] = np.nan
    return out, names, np.array(swap)


def attack_direction(match, f):
    """Per frame: +1 if the home team attacks towards x = length, -1 otherwise, 0 outside play.

    Read off the data, one value per half: averaged over the whole half, the side whose players
    stand at smaller x defends the x = 0 goal.
    """
    out = np.zeros(f.n, dtype=np.float32)
    x = f.pos[..., 0]
    for start, stop in match.halves:
        window = slice(start, stop, 5)
        with np.errstate(invalid="ignore"):
            home = np.nanmean(x[window][:, match.team == 0])
            away = np.nanmean(x[window][:, match.team == 1])
        out[start:stop] = 1.0 if home < away else -1.0
    return out


SHAPE_LAGS_S = (-5, -2, 2, 5)


def _row_quantile(sorted_rows, count, q):
    """Quantile q of the first `count` entries of each row (rows sorted, padding at the end)."""
    position = q * (np.maximum(count, 1) - 1)
    low = np.floor(position).astype(np.int64)
    high = np.minimum(low + 1, np.maximum(count, 1) - 1)
    weight = position - low
    rows = np.arange(len(sorted_rows))
    a, b = sorted_rows[rows, low], sorted_rows[rows, high]
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(a) & np.isfinite(b), a + weight * (b - a), np.nan)


def shape_view(f, match):
    """(features (N, k), names, swap_index): who is attacking, judged from the players alone.

    Used only on frames where the ball is not tracked. Every quantity is expressed in the frame of
    the team it belongs to ("own attacking direction is positive"), so exchanging the two teams
    is a pure column swap and the model can be made exactly symmetric.
    """
    fps = f.fps
    length = f.pitch[0]
    direction = attack_direction(match, f)
    u_all = (f.pos[..., 0] - length / 2.0) * direction[:, None]     # home attacks towards +u
    per_team = []
    for side, sign in ((0, 1.0), (1, -1.0)):
        cols = match.team == side
        u = sign * u_all[:, cols]
        y = f.pos[:, cols, 1]
        here = ~np.isnan(u)
        k = np.maximum(here.sum(axis=1), 1)
        enough = here.sum(axis=1) >= 5
        # Order statistics rather than means: the automatic tracking often lacks the goalkeeper
        # the camera cannot see, and a mean or a standard deviation moves when one player
        # appears or vanishes. Quantiles over the players that are there barely do.
        su = np.sort(np.where(here, u, np.inf), axis=1)
        sy = np.sort(np.where(here, y, np.inf), axis=1)
        q25, q50, q75 = (_row_quantile(su, k, q) for q in (0.25, 0.5, 0.75))
        y25, y75 = (_row_quantile(sy, k, q) for q in (0.25, 0.75))
        advance = np.where(enough, q50, np.nan)
        spread_u = np.where(enough, q75 - q25, np.nan)
        spread_iqr = np.where(enough, y75 - y25, np.nan)
        # Width of the team, without its goalkeeper: with all eleven tracked the deepest player
        # is left out, so the number means the same whether or not the keeper is in view.
        field = here.copy()
        full = here.sum(axis=1) >= 11
        deepest = np.argmin(np.where(here, u, np.inf), axis=1)
        field[np.flatnonzero(full), deepest[full]] = False
        fk = np.maximum(field.sum(axis=1), 1)
        yz = np.where(field, y, 0.0)
        my = yz.sum(axis=1) / fk
        spread_y = np.where(enough, np.sqrt(np.sum(field * (yz - my[:, None]) ** 2, axis=1) / fk), np.nan)
        back = np.where(enough, su[:, 2], np.nan)           # third deepest: a defender either way
        front = np.where(enough, np.max(np.where(here, u, -np.inf), axis=1), np.nan)
        speed = f.speed[:, cols]
        sk = np.maximum((~np.isnan(speed)).sum(axis=1), 1)
        mean_speed = np.where(enough, np.nansum(speed, axis=1) / sk, np.nan)
        # Mean of the players' own velocities along the team's attacking direction. Unlike the
        # velocity of the centroid, it does not jump when a track starts or ends.
        vx = sign * direction[:, None] * f.vel[:, cols, 0]
        vk = np.maximum((~np.isnan(vx)).sum(axis=1), 1)
        vel = np.where(enough, np.nansum(vx, axis=1) / vk, np.nan)
        v1 = moving_mean(vel, fps // 2)
        v3 = moving_mean(vel, 3 * fps // 2)
        cols_out = [("advance", advance), ("v1", v1), ("v3", v3), ("back", back), ("front", front),
                    ("spread_u", spread_u), ("spread_y", moving_mean(spread_y, fps // 2)),
                    ("spread_y_iqr", moving_mean(spread_iqr, fps // 2)),
                    ("speed", moving_mean(mean_speed, fps // 2))]
        for lag_s in SHAPE_LAGS_S:
            cols_out.append(("v1@{:+g}s".format(lag_s), shift(v1, int(lag_s * fps))))
        per_team.append(cols_out)
    columns, names, swap = [], [], []
    for (name, home), (_, away) in zip(per_team[0], per_team[1]):
        swap += [len(columns) + 1, len(columns)]
        columns += [home, away]
        names += [name + "_H", name + "_A"]
    out = np.stack(columns, axis=1).astype(np.float32)
    out[direction == 0] = np.nan
    return out, names, np.array(swap)


PLAYER_LAGS_S = (-10, -6, -3, -1.5, 1.5, 3, 6, 10)


def players_view2(f, match):
    """(features (N, k), names). Rank-normalised within the match; never looks at the ball."""
    fps = f.fps
    speed, pos = f.speed, f.pos
    present = ~np.isnan(speed)
    count = np.maximum(present.sum(axis=1), 1)
    enough = present.sum(axis=1) > 3

    def per_frame(values):
        return np.where(enough, values, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = per_frame(np.nansum(speed, axis=1) / count)
        slow = per_frame(np.sum(speed < 1.0, axis=1) / count)
        jog = per_frame(np.sum(speed > 3.0, axis=1) / count)
        sprint = per_frame(np.sum(speed > 5.0, axis=1) / count)
        top = per_frame(np.max(np.where(present, speed, -1.0), axis=1))
        team_means = []
        for side in (0, 1):
            sub = speed[:, match.team == side]
            cnt = np.maximum((~np.isnan(sub)).sum(axis=1), 1)
            team_means.append(np.nansum(sub, axis=1) / cnt)
        lo = per_frame(np.minimum(team_means[0], team_means[1]))
        hi = per_frame(np.maximum(team_means[0], team_means[1]))
        accel = np.full(speed.shape, np.nan, dtype=np.float32)
        accel[2:-2] = np.abs(speed[4:] - speed[:-4]) * (fps / 4.0)
        acc = per_frame(np.nansum(accel, axis=1) / np.maximum((~np.isnan(accel)).sum(axis=1), 1))
        px = np.where(np.isnan(pos[..., 0]), 0.0, pos[..., 0])
        py = np.where(np.isnan(pos[..., 1]), 0.0, pos[..., 1])
        here = ~np.isnan(pos[..., 0])
        k = np.maximum(here.sum(axis=1), 1)
        mx, my = px.sum(axis=1) / k, py.sum(axis=1) / k
        spread_x = per_frame(np.sqrt(np.sum(here * (px - mx[:, None]) ** 2, axis=1) / k))
        spread_y = per_frame(np.sqrt(np.sum(here * (py - my[:, None]) ** 2, axis=1) / k))

    half = fps // 2
    series = [
        ("m1", moving_mean(mean, half)), ("m3", moving_mean(mean, 3 * half)),
        ("m8", moving_mean(mean, 8 * half)), ("m20", moving_mean(mean, 20 * half)),
        ("slow1", moving_mean(slow, half)), ("jog1", moving_mean(jog, half)),
        ("sprint2", moving_mean(sprint, fps)), ("top1", moving_mean(top, half)),
        ("team_lo2", moving_mean(lo, fps)), ("team_hi2", moving_mean(hi, fps)),
        ("accel2", moving_mean(acc, fps)),
        ("spread_x2", moving_mean(spread_x, fps)), ("spread_y2", moving_mean(spread_y, fps)),
    ]
    columns, names = [], []
    ranked = {}
    for name, values in series:
        ranked[name] = quantile_normalise(values, f.graded)
        columns.append(ranked[name])
        names.append(name)
    for lag_s in PLAYER_LAGS_S:
        lag = int(round(lag_s * fps))
        for name in ("m1", "slow1"):
            columns.append(shift(ranked[name], lag))
            names.append("{}@{:+g}s".format(name, lag_s))
    for future in (False, True):
        tag = "next5s" if future else "last5s"
        columns.append(_rolling(ranked["m1"], 5 * fps, "max", future))
        names.append("m1_max_" + tag)
        columns.append(_rolling(ranked["m1"], 5 * fps, "min", future))
        names.append("m1_min_" + tag)
    return np.stack(columns, axis=1).astype(np.float32), names
