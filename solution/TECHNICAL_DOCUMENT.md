# Technical document — ball possession from automatic tracking

More than a Hack 2027, Phase 1. Code: `solution/` in this repository. Run: `python solution/run.py`.

## 1. Summary

Every graded frame is labelled `H`, `A` or `D` by a pipeline with three learned evidence models
and one decoder. Nothing in it was fitted, tuned or hand-labelled on the competition match:

- the models are trained on **seven other matches** with public frame-level labels;
- the decoder's eight hyper-parameters are chosen on those matches with the organisers' scorer;
- the only things learned on the competition match are learned **without labels**.

| check | spellF1 | macroF1 |
|---|---|---|
| External validation, 3 Metrica sample games, leave-one-match-out | **0.589** | 0.860 |
| — organisers' baseline (±50 frames) on the same input | 0.321 | 0.746 |
| External validation, 4 DFL matches, leave-one-match-out | **0.512** | 0.858 |
| — organisers' baseline (±50 frames) on the same input | 0.282 | 0.735 |
| Public 5 minutes of the competition match (pipeline check only, 17 possessions) | 0.676 | 0.904 |

The submitted file has 295 possessions (mean 11.6 s) and 87 stoppages; 46.0% of graded frames are
`D`, 33.2% `H`, 20.8% `A`. No graded frame is `X`.

## 2. Approach

The task is treated as **segmentation**, because the ranking metric counts possessions, not
frames. Two questions are answered separately and then decoded together:

1. *Is the ball in play?* — from how the players move.
2. *If so, which team has it?* — from the ball where it is tracked, from the teams' shapes where
   it is not.

```
ATD ──► features ──► alive model  ─┐
                 ──► team model    ├─► per-frame evidence ─► Viterbi (H/A/D) ─► minimum durations ─► labels
                 ──► shape model  ─┤
ball track ──► ball-state evidence ┤   (meaning learned on this match, no labels)
Smart Tagging ──► priors ──────────┘
```

### 2.1 Three views of a frame (`mtah/features.py`)

- **Players view** (33 features). Mean player speed over 1, 3, 8 and 20 s; share of players
  walking, jogging, sprinting; top speed; the slower and the faster team; mean acceleration;
  spread of all players; the 1-s mean speed 1.5–10 s before and after; its minimum and maximum
  over the previous and next 5 s. Every feature is **rank-normalised within the match**, so a
  model fitted on one tracking system transfers to another. It never looks at the ball, so it
  exists on all frames.
- **Ball view** (45 features). Distance from the ball to the nearest player of each team at 11
  offsets from −2 s to +2 s; its minimum over the previous and next second; time since / until
  each team was within 1.5 m; time since / until a touch by each team (a local peak of ball
  velocity change with a player within 2.5 m); relative speed of the nearest player; ball speed
  and velocity change. Defined only where the ball is tracked.
- **Shape view** (26 features). Per team, in its own attacking direction: width (spread across
  the pitch, goalkeeper excluded), depth, median advance, back and front line, mean player speed,
  mean velocity towards the opponent's goal now and ±2 s, ±5 s. The team with the ball spreads
  out and the team without it narrows; width alone carries over half of this model's gain.
  Attack direction is read from the data (the side whose players stand at smaller x over the
  half), and agrees with the ATD metadata.

### 2.2 Three models (`mtah/gbm.py`)

Gradient-boosted trees (LightGBM, binary log-loss, 31 leaves, learning rate 0.05, fixed seed,
single thread, deterministic mode).

| model | predicts | input | used on |
|---|---|---|---|
| alive (300 trees) | ball in play | players view | every frame |
| team (400 trees) | home has the ball | ball view + shape view | frames with a tracked ball |
| shape (200 trees) | home has the ball | shape view | frames without one |

The two possession models are **symmetric by construction**: each training example is also shown
with the teams exchanged, and a prediction is the average of the two orientations. Neither can
prefer a side.

Leave-one-match-out frame accuracy on the degraded external matches: alive 0.88–0.94; team
0.88–0.95 where the ball is tracked; shape 0.82–0.90 where it is not.

### 2.3 Ball-state evidence, learned on the match without labels (`mtah/model.py`)

What the ball track says about alive-versus-dead cannot be learned from other matches: the
external data either blanks or freezes the ball during stoppages, while the ATD shows it rolling
away, being carried to the spot, or not at all. So each frame's ball track is reduced to one of
eleven coarse categories, and the meaning of each category is **estimated on the competition
match itself**: the alive model's probability is averaged over all frames of the category, and
the log-odds of that average against the match as a whole becomes the category's evidence. One
pass, no feedback loop, no labels.

| ball category | share of graded frames | estimated P(in play) | evidence (log-odds) |
|---|---|---|---|
| tracked outside the pitch | 1.6% | 0.43 | −0.58 |
| still | 2.1% | 0.22 | −1.58 |
| slow, a player on it | 9.4% | 0.68 | +0.48 |
| slow, nobody near | 3.6% | 0.58 | +0.03 |
| moving | 24.3% | 0.88 | +1.71 |
| fast | 18.4% | 0.90 | +1.96 |
| lost, run shorter than 8 s | 3.8% | 0.64 | +0.29 |
| lost, 8–15 s | 6.4% | 0.40 | −0.68 |
| lost, 15–30 s | 12.4% | 0.16 | −1.91 |
| lost, 30–60 s | 10.4% | 0.16 | −1.93 |
| lost, over 60 s | 7.7% | 0.13 | −2.22 |

### 2.4 Decoder (`mtah/pipeline.py`, `mtah/model.py`)

A three-state Viterbi path over `H`, `A`, `D`, run inside each half. Per frame, "alive" scores
+½·a and "dead" −½·a, where a is the tempered sum of the alive model's log-odds and the ball-state
evidence. The team evidence only splits "alive" between the two teams (the favoured team scores
what "alive" scores, the other is penalised), so a confident possession call can never make a
frame look more alive. Switching state costs a fixed amount. Afterwards, stoppages shorter than
4 s and possessions shorter than 0.32 s are absorbed into their neighbours.

| parameter | value | | parameter | value |
|---|---|---|---|---|
| alive temperature | 0.06 | | cost alive → dead | 6.0 |
| team temperature (ball tracked) | 0.25 | | cost dead → alive | 6.0 |
| shape temperature (ball lost) | 0.10 | | cost of a turnover | 1.5 |
| minimum stoppage | 100 frames | | minimum possession | 8 frames |

### 2.5 Priors for the competition match (`mtah/priors.py`)

All read by code from the provided assets; none is tied to a hand-picked frame.

- **A half is dead until its kick-off.** The kick-off instant is the frame, inside the tagged
  `Kick Off` window, where the tracker re-acquires the ball in the centre circle; where that
  cannot be seen, the median offset of the kick-offs that could be seen in this match (82 frames
  after the window opens). This removes the 2,934 graded frames before the second-half kick-off.
- **`GOALS` tags** run from the goal to the following kick-off: soft dead-ball evidence across
  the interval, and the conceding side is favoured at the restart.
- **Other `SET PIECES` tags**: dead-ball evidence for the first second of the window, and a
  small preference for the tagged team (when it is not `N/A`) over the next 15 s.
- **Ball out of play**: where the tracked ball crosses a boundary line, gets at least 0.5 m
  outside, stays out for half a second (or the track ends there) and is not moving at
  ball-in-flight speed, the frames outside are dead-ball evidence. 45 such episodes.

These priors change 2.5% of graded frames. They could not be validated externally (no other
match has Smart Tagging), so they are few, mostly soft, and fixed a priori.

## 3. Assets: used and dropped

| asset | status | why |
|---|---|---|
| Tactical ATD (players, ball, team membership) | **used** | the whole pipeline |
| Smart Tagging | **used, 3 codes** | `SET PIECES` (incl. kick-offs) and `GOALS` as priors |
| Smart Tagging phase codes (Build Up, Progression, …) | dropped | 15–250 s templated windows, half are mirror pairs; no way to validate a weight for them on other matches |
| ATD provider speed | dropped | recomputed from positions, so features mean the same on every dataset |
| Tactical homography | dropped | not needed once working in pitch coordinates |
| Tactical / broadcast video | dropped | no detector could be validated in the time available |
| Broadcast ATD + homography | dropped | it aligns to the tactical feed with one constant offset per half, but its ball is tracked on only ~12% of the frames where the tactical ball is lost |

## 4. Training data

No data from the competition match was used for training or validation, and nothing was labelled
by us. Two public datasets of **other games**, 7 matches, 1,000,876 frames (11.1 h), 576,804 in
play:

| dataset | matches | labels, and who made them | source |
|---|---|---|---|
| Metrica Sports sample data | 3 anonymised matches | corrected tracking and events produced and validated by Metrica Sports. Frame labels derived by `tools/metrica_to_arrays.py`: `D` where the corrected ball is blank for ≥ 1 s; otherwise the team of the latest on-ball event (pass, carry, shot, recovery, set piece, ball lost, ball out) | github.com/metrica-sports/sample-data |
| Open DFL dataset (Bassek et al., *Sci Data* 12, 195, 2025; CC-BY 4.0) | 4 of its 7 matches (J03WMX, J03WN1, J03WOH, J03WPY) | official DFL position data: per-frame `BallStatus` (in play) and `BallPossession` (home / guest), produced by the league's data provider. Converted by `tools/dfl_to_arrays.py` | huggingface.co/datasets/pysport/idsse-data (kloppy mirror) |

**Making clean data look like the ATD** (`mtah/simulate.py`). The external tracking is far
cleaner than the ATD, so before training or validating, every external match is degraded:

- the ball is removed during every stoppage and during random runs of 5–60 s of open play
  (mean 45 s of visibility between runs for training; 45 s and 120 s for validation);
- the ball track gets a slow drift error (0.55 m, 2 s correlation), a Gaussian low-pass filter
  (σ = 6 frames) and 3 mm jitter. These four numbers were chosen by `tools/fit_degradation.py`
  to match **label-free** statistics of the ATD ball track: quantiles of its velocity change
  (99th percentile 2.9 m/s against ~17 m/s in clean data), of its high-frequency residual, and of
  its distance to the nearest player (median 1.9 m against 0.3–1.4 m).

## 5. Noisy and missing tracking

- **Ball lost (41% of graded frames).** Alive / dead is still decided, by the players view plus
  the "lost for N seconds" category. Possession is decided by the shape model, which never needs
  the ball. No frame is left to a default.
- **Anonymous, fragmented tracks.** No feature follows an individual. Velocities are central
  differences within a slot; a step implying more than 13 m/s is treated as an identity switch
  and blanked.
- **Missing players.** Shape features are order statistics or exclude the deepest player, so they
  barely move when a player — typically the far goalkeeper — is absent. Slots 24 and 46 carry
  inferred goalkeeper positions when the keeper is out of view; they are kept, because the models
  were trained with both keepers present.
- **Ball errors.** The ball view is trained on a ball degraded to the ATD's own noise level, so
  it does not rely on centimetre proximity or sharp kicks. A tracked ball outside the pitch only
  counts as out of play after the checks in §2.5.
- **Flicker.** Removed by the decoder (switch costs, minimum durations), not by per-frame voting.

## 6. Possession rules implemented

- **Three states only**; a possession is a maximal run of one team with the ball alive.
- **Loose and contested balls stay with the last team.** There is no "contested" state; with no
  clear evidence the decoder keeps its state, and a turnover has to pay a switch cost.
- **Touches and control are learned, not thresholded.** The team model is trained on labels in
  which possession moves at the new team's first on-ball event, so it reproduces that convention
  as far as a low-pass-filtered ball allows. Changes shorter than 0.32 s are never reported.
- **Dead ball** starts where players stop and/or the ball leaves the pitch; never shorter than
  4 s. **Dead-ball ownership is not predicted**: the label is `D`.
- **Restarts.** The first team after a stoppage comes from the evidence at the restart, with a
  small preference for the tagged team at tagged set pieces and for the conceding team after a
  goal.
- **`X`** only outside the two halves. On graded frames an abstention is scored as a repeat of
  the previous label, so the decoder commits instead.

## 7. Validation, and what the public five minutes were used for

- **Hyper-parameters** were chosen by `tools/tune.py`: coordinate descent from three starting
  points on a fixed grid, maximising the official spellF1 averaged over the three Metrica games
  at two ball-loss levels, on **leave-one-match-out** predictions. The three starts converge to
  the same score (0.587–0.589). DFL matches are reported, not optimised.
- While tuning, the ball-state evidence is switched off: in the degraded data the ball is hidden
  exactly when dead, which would make it unrealistically informative.
- Training the possession models on Metrica games only (closer in label convention to the
  competition) scored lower (0.574) than training on all seven, so all seven are used.
- **The public 5-minute ground truth was never used to fit or select anything.** The pipeline was
  scored on it to check frame alignment, file format and label meaning. One such check exposed a
  defect in the decoder (confident possession evidence was leaking into the alive / dead
  decision); the defect was fixed on principle and the hyper-parameters were re-tuned on the
  external matches only.
- Per-match external results: `python solution/tools/report.py`.

## 8. Limitations

- **Short possessions are under-reported.** On external matches the pipeline reports about two
  thirds of the true number of possessions; spells under ~2 s are mostly missed.
- **Stoppages that look like play are missed** (a quick throw-in while players keep running),
  and a stoppage called by whistle is detected a second or two late.
- **The priors are unvalidated**, and the ball-state evidence rests on the alive model being
  roughly calibrated on this match.
- **The external labels are not the competition's labels.** DFL moves possession on more touches
  than the public ground truth appears to; Metrica's sample labels are derived by us from events.
- **Simulated degradation is not the real thing.** Real ball loss is correlated with what is
  happening on the pitch; the simulated loss is random.
- With more time: a ball detector on the tactical video for the blackouts; whistle detection
  from the audio track for stoppages; player-dropout augmentation; a duration-aware (semi-Markov)
  decoder.

## 9. Reproduction

```bash
pip install -r solution/requirements.txt      # numpy, pandas, lightgbm, scipy - pinned
python solution/run.py --team "<team name>"   # writes solution/output/<team name>.csv
```

Python 3.12. Under a minute on a laptop CPU. The run is deterministic and the file is validated
with the organisers' `validate_submission` checks before the script exits. Retraining
(`solution/README.md`) reproduces the committed model files byte for byte.
