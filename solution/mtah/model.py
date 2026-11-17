"""The statistical core: a three-state Viterbi decoder with minimum durations, and the
label-free adaptation of the ball-state evidence to the match being labelled.

numpy only. Nothing in this file is random, so the same inputs always give the same labels.
"""
import numpy as np

H, A, D = 0, 1, 2


# -- ball-state categories ----------------------------------------------------------------------
#
# A coarse description of what the ball track is doing on each frame. What each category says
# about alive-versus-dead is NOT assumed: it is estimated on the match being labelled, without
# labels, from the agreement between this view and the players view (see adapt_ball_evidence).

GAP_BINS_S = (8.0, 15.0, 30.0, 60.0)          # missing-ball runs, by total length in seconds
CATEGORY_NAMES = (
    "ball out of the pitch", "ball still", "ball slow, player on it", "ball slow, nobody near",
    "ball moving", "ball fast",
    "lost < 8 s", "lost 8-15 s", "lost 15-30 s", "lost 30-60 s", "lost > 60 s")
N_CATEGORIES = len(CATEGORY_NAMES)


def run_lengths(codes):
    """For each position, the length of the constant run it sits in."""
    n = len(codes)
    breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [n]))
    lengths = np.diff(bounds)
    return np.repeat(lengths, lengths)


def ball_categories(f, out_margin=1.0):
    """(category per frame, run length per frame). Frames feature object from features.build."""
    from .features import moving_mean
    n, fps = f.n, f.fps
    category = np.zeros(n, dtype=np.int8)
    speed = moving_mean(f.ball_speed, fps // 2)
    speed = np.where(np.isnan(speed), 0.0, speed)
    nearest = np.minimum(f.near_dist[:, 0], f.near_dist[:, 1])
    visible = f.visible
    with np.errstate(invalid="ignore"):
        outside = visible & (f.out_dist > out_margin)
    category[visible] = 4
    category[visible & (speed >= 8.0)] = 5
    category[visible & (speed < 3.0) & (nearest < 2.0)] = 2
    category[visible & (speed < 3.0) & (nearest >= 2.0)] = 3
    category[visible & (speed < 0.5)] = 1
    category[outside] = 0
    gap = run_lengths(visible.astype(np.int8)) / float(fps)
    missing = ~visible
    category[missing] = 6 + np.searchsorted(np.array(GAP_BINS_S), gap[missing], side="right")
    return category, run_lengths(category)


def forward_backward(log_odds_alive, stay_alive, stay_dead, blocks):
    """Posterior P(alive) per frame of a two-state HMM, run separately inside each block."""
    n = len(log_odds_alive)
    gamma = np.full(n, 0.5)
    paa, pad = stay_alive, 1.0 - stay_alive
    pdd, pda = stay_dead, 1.0 - stay_dead
    for start, stop in blocks:
        odds = np.clip(log_odds_alive[start:stop], -30, 30)
        ea = (1.0 / (1.0 + np.exp(-odds))).tolist()
        ed = (1.0 / (1.0 + np.exp(odds))).tolist()
        m = stop - start
        fa, fd = [0.0] * m, [0.0] * m
        xa, xd = 0.5 * ea[0], 0.5 * ed[0]
        total = xa + xd
        fa[0], fd[0] = xa / total, xd / total
        for t in range(1, m):
            ya = (fa[t - 1] * paa + fd[t - 1] * pda) * ea[t]
            yd = (fa[t - 1] * pad + fd[t - 1] * pdd) * ed[t]
            total = ya + yd
            fa[t], fd[t] = ya / total, yd / total
        ba, bd = 1.0, 1.0
        out = [0.0] * m
        out[m - 1] = fa[m - 1]
        for t in range(m - 2, -1, -1):
            wa, wd = ea[t + 1] * ba, ed[t + 1] * bd
            na = paa * wa + pad * wd
            nd = pda * wa + pdd * wd
            total = na + nd
            ba, bd = na / total, nd / total
            ga, gd = fa[t] * ba, fd[t] * bd
            out[t] = ga / (ga + gd)
        gamma[start:stop] = out
    return gamma


def adapt_ball_evidence(players_logit, category, f, blocks, params):
    """Per-category log-odds of alive, learned on this match with no labels.

    The players view gives a calibrated probability that the ball is in play on every frame.
    Averaging that probability over all frames of one ball category estimates what the category
    means in THIS tracking feed - how often a tracked ball that is moving really is in play, how
    often a 20-second loss of the ball is a stoppage rather than a crowded box. The log-odds of
    that average, relative to the match as a whole, is the evidence the category then
    contributes on each of its frames.

    With em_iterations > 0 the estimate is refined by expectation-maximisation, re-averaging the
    smoothed posterior of the combined model. The default is 0: one pass of plain averaging, which
    has no feedback loop and so cannot talk itself into an extreme.
    """
    graded = f.graded
    temp = params["em_temperature"]
    stay_alive = 1.0 - 1.0 / (params["em_alive_s"] * f.fps)
    stay_dead = 1.0 - 1.0 / (params["em_dead_s"] * f.fps)
    counts = np.bincount(category[graded], minlength=N_CATEGORIES).astype(np.float64)
    pseudo = 10.0 * f.fps        # shrink towards the match prior with ten seconds of frames
    gamma = 1.0 / (1.0 + np.exp(-np.clip(players_logit, -30, 30)))
    lam = np.zeros(N_CATEGORIES)
    share = np.full(N_CATEGORIES, 0.5)
    for iteration in range(params["em_iterations"] + 1):
        prior = gamma[graded].mean()
        alive_mass = np.bincount(category[graded], weights=gamma[graded], minlength=N_CATEGORIES)
        share = np.clip((alive_mass + pseudo * prior) / (counts + pseudo), 1e-3, 1 - 1e-3)
        lam = np.clip(np.log(share / (1 - share)) - np.log(prior / (1 - prior)),
                      -params["em_clip"], params["em_clip"])
        if iteration < params["em_iterations"]:
            gamma = forward_backward(temp * (players_logit + lam[category]), stay_alive,
                                     stay_dead, blocks)
    return lam, share, counts


# -- three-state decoding -----------------------------------------------------------------------

def viterbi(score, cost, blocks):
    """Most likely H/A/D path. score (N, 3) is added per frame, cost[i][j] is paid on i -> j."""
    n = len(score)
    path = np.full(n, D, dtype=np.int8)
    c = [[float(cost[i][j]) for j in range(3)] for i in range(3)]
    for start, stop in blocks:
        m = stop - start
        s = score[start:stop].tolist()
        back = np.empty((m, 3), dtype=np.int8)
        v0, v1, v2 = s[0]
        for t in range(1, m):
            e0, e1, e2 = s[t]
            # into state 0
            a, b, d = v0, v1 - c[1][0], v2 - c[2][0]
            if a >= b and a >= d:
                n0, k0 = a, 0
            elif b >= d:
                n0, k0 = b, 1
            else:
                n0, k0 = d, 2
            a, b, d = v0 - c[0][1], v1, v2 - c[2][1]
            if b >= a and b >= d:
                n1, k1 = b, 1
            elif a >= d:
                n1, k1 = a, 0
            else:
                n1, k1 = d, 2
            a, b, d = v0 - c[0][2], v1 - c[1][2], v2
            if d >= a and d >= b:
                n2, k2 = d, 2
            elif a >= b:
                n2, k2 = a, 0
            else:
                n2, k2 = b, 1
            v0, v1, v2 = n0 + e0, n1 + e1, n2 + e2
            back[t, 0], back[t, 1], back[t, 2] = k0, k1, k2
        state = int(np.argmax([v0, v1, v2]))
        out = np.empty(m, dtype=np.int8)
        for t in range(m - 1, -1, -1):
            out[t] = state
            state = back[t, state] if t > 0 else state
        path[start:stop] = out
    return path


def segments(codes, start, stop):
    window = codes[start:stop]
    breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(window)]))
    return [(start + int(bounds[i]), start + int(bounds[i + 1]), int(window[bounds[i]]))
            for i in range(len(bounds) - 1)]


def enforce_minimum_durations(path, blocks, min_dead, min_spell, team_score):
    """Remove segments too short to be real, in place.

    A dead ball shorter than min_dead frames is absorbed into its neighbours; a possession shorter
    than min_spell frames is absorbed likewise. An absorbed segment takes the label of the
    neighbour the frame-level team evidence prefers when its two neighbours disagree.
    """
    for start, stop in blocks:
        for label_kind, minimum in (("dead", min_dead), ("spell", min_spell)):
            changed = True
            while changed:
                changed = False
                segs = segments(path, start, stop)
                for i, (a, b, k) in enumerate(segs):
                    is_dead = k == D
                    if (label_kind == "dead") != is_dead or b - a >= minimum:
                        continue
                    left = segs[i - 1][2] if i > 0 else None
                    right = segs[i + 1][2] if i + 1 < len(segs) else None
                    if left is None and right is None:
                        continue
                    if left is None or right is None or left == right:
                        fill = left if left is not None else right
                    elif D in (left, right) and label_kind == "spell":
                        # between a stoppage and the other team: give it to the other team
                        fill = left if left != D else right
                    elif label_kind == "dead":
                        # dead blip between H and A: split it at the middle
                        mid = (a + b) // 2
                        path[a:mid] = left
                        path[mid:b] = right
                        changed = True
                        break
                    else:
                        fill = left
                    path[a:b] = fill
                    changed = True
                    break
    return path
