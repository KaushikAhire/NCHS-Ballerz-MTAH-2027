"""Regenerate the submission from the provided data. One command, no arguments needed:

    python solution/run.py                     # -> solution/output/submission.csv
    python solution/run.py --team "My Team"    # -> solution/output/My Team.csv  (+ My Team.md)

Reads the tactical ATD and the Smart Tagging from ./data (or MTAH_DATA_DIR), loads the two
models committed in solution/models, and writes one label per video frame. Deterministic: the
same inputs give a byte-identical file.
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "evaluation"))

from mtah import config, gbm, io_match, pipeline, priors  # noqa: E402
import possession_metrics as pm  # noqa: E402

LETTERS = np.array(["H", "A", "D", "X"])


def label_competition_match(data_dir=None, verbose=True):
    """(labels as letters for every video frame, diagnostics dict)."""
    started = time.time()
    match = io_match.load_match(data_dir)
    models = gbm.load()
    params = dict(pipeline.DEFAULT_PARAMS, **config.DECODER)
    ev = pipeline.evidence(match, models, params)
    alive_prior, team_prior, notes = priors.build(match, ev["frames"], config.PRIORS)
    codes = pipeline.decode(ev, params, alive_prior, team_prior)

    labels = np.full(match.n, "X", dtype="<U1")      # ungraded frames: no answer
    for start, stop in match.halves:
        labels[start:stop] = LETTERS[codes[start:stop]]
    if verbose:
        graded = ev["frames"].graded
        share = np.bincount(codes[graded], minlength=3) / float(graded.sum())
        stats = pm.possession_stats(codes, match.halves)
        print("labelled {} frames in {:.0f} s".format(match.n, time.time() - started))
        print("  share of graded frames   H {:.3f}  A {:.3f}  D {:.3f}".format(*share))
        print("  possessions {}  mean {:.1f} s  home share of live time {:.3f}".format(
            stats["n_spells"], stats["mean_duration_s"], stats["home_share"]))
        print("  kick-offs at {}; {} tracked ball-out episodes".format(
            notes["kickoffs"], notes["out_of_play"]))
    return labels, {"evidence": ev, "notes": notes, "codes": codes, "match": match}


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default="submission",
                        help="team name; the file is written as <team>.csv")
    parser.add_argument("--out", default=os.path.join(HERE, "output"), help="output folder")
    parser.add_argument("--data", default=None, help="data folder (default ./data)")
    args = parser.parse_args()

    labels, _ = label_competition_match(args.data)
    path = write_submission(os.path.join(args.out, args.team + ".csv"), labels)

    # The organisers' own checks, on the exact file just written.
    reason = pm.validate_submission(pm.read_label_csv(path), io_match.N_FRAMES)
    print("wrote {}".format(path))
    # The technical document, named after the team as the submission form asks.
    document = os.path.join(HERE, "TECHNICAL_DOCUMENT.md")
    if os.path.isfile(document):
        shutil.copyfile(document, os.path.join(args.out, args.team + ".md"))
        print("wrote {}".format(os.path.join(args.out, args.team + ".md")))
    print("validation: {}".format("OK" if reason is None else "FAILED - " + reason))
    sys.exit(0 if reason is None else 1)
