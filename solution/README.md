# Ball possession from automatic tracking — More than a Hack 2027, Phase 1

One label (`H` / `A` / `D` / `X`) for each of the 199,319 frames of the tactical video of
FC Barcelona – Athletic Club, produced entirely by an algorithm from the provided automatic
tracking (ATD) and Smart Tagging.

The method, the choices behind it and its limits are in
**[TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md)**. This page is how to run it.

## Regenerate the submission

```bash
pip install -r solution/requirements.txt
python solution/run.py --team "<team name>"
```

That reads `data/tactical/` (the provided bundle; `MTAH_DATA_DIR` moves it), loads the three
models committed in `solution/models/`, and writes `solution/output/<team name>.csv`. It takes
under a minute on a laptop CPU, runs the organisers' own validation on the file it wrote, and is
deterministic: the same inputs give a byte-identical file.

Only the tactical ATD (`*_atd.xml`, `*_atd.txt`) and the Smart Tagging XML are read. File names
are matched by pattern, so the bundle works with the names in `DATA.md` or the names the download
arrived with.

## What is in here

```
solution/
  run.py                    the one command that regenerates the submission
  requirements.txt          pinned dependencies
  TECHNICAL_DOCUMENT.md     approach, assets, training data, rules implemented, limitations
  models/                   the three trained models (LightGBM text files) + training metadata
  mtah/
    io_match.py             read the ATD and the Smart Tagging into arrays
    features.py             the players view, the ball view and the team-shape view
    gbm.py                  the three evidence models: alive, team (ball tracked), shape (ball lost)
    model.py                label-free adaptation of the ball-state evidence; Viterbi decoder
    priors.py               Smart Tagging and ball-out-of-pitch evidence for the competition match
    pipeline.py             evidence -> H / A / D
    config.py               the hyper-parameters the submission was produced with
    simulate.py             degrade clean external tracking to look like the ATD; scoring helper
    external.py             load the converted external matches
  tools/                    everything used to BUILD the models - not needed to run them
    download_external.py    fetch the public external matches (other games)
    metrica_to_arrays.py    Metrica sample games -> arrays with frame labels
    dfl_to_arrays.py        open DFL dataset -> arrays with frame labels
    fit_degradation.py      choose the ball-noise model from label-free statistics of the ATD
    train.py                fit the three models on the external matches
    oof.py                  leave-one-match-out predictions on the external matches
    tune.py                 choose the decoder's hyper-parameters with the official scorer
    report.py               external validation table against the organisers' baseline
```

Everything outside `solution/` is the organisers' repository, unchanged apart from a few lines
added to `.gitignore` and a pointer at the top of the root README. Their scorer (`evaluation/possession_metrics.py`) is imported, not copied.

## Rebuild the models from scratch (optional)

Not needed to reproduce the submission. It needs about 1.7 GB of public data from other games.

```bash
python solution/tools/download_external.py
python solution/tools/metrica_to_arrays.py
python solution/tools/dfl_to_arrays.py
python solution/tools/fit_degradation.py     # prints the ball-noise parameters in mtah/simulate.py
python solution/tools/train.py               # rewrites solution/models/
python solution/tools/tune.py                # prints the decoder parameters in mtah/config.py
python solution/tools/report.py              # external validation table
```

## Data

No match data is in this repository. `data/`, `external/`, `solution/cache/` and
`solution/output/` are git-ignored: the competition data is not redistributable, and a submission
file is a full set of labels for the match.

## Licence

Code in `solution/` is released under the MIT licence, like the repository it sits in.
