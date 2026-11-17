"""Download the public external matches used for training and validation (about 1.7 GB).

Only needed to RETRAIN the models or to re-run the external validation. Regenerating the
submission (solution/run.py) needs none of this: the trained models are committed.

  * Metrica Sports sample data, games 1-3 - github.com/metrica-sports/sample-data
  * Open DFL dataset, 4 of its 7 matches   - Bassek et al., "An integrated dataset of
    spatiotemporal and event data in elite soccer", Scientific Data 12, 195 (2025), CC-BY 4.0,
    served from the kloppy project's mirror (huggingface.co/datasets/pysport/idsse-data)

None of these is the competition match.

    python solution/tools/download_external.py
    python solution/tools/metrica_to_arrays.py
    python solution/tools/dfl_to_arrays.py
"""
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
EXTERNAL = os.path.join(os.path.dirname(os.path.dirname(HERE)), "external")

METRICA = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
DFL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
DFL_MATCHES = ("DFL-COM-000001_DFL-MAT-J03WMX", "DFL-COM-000001_DFL-MAT-J03WN1",
               "DFL-COM-000002_DFL-MAT-J03WOH", "DFL-COM-000002_DFL-MAT-J03WPY")


def files():
    for game in (1, 2):
        for part in ("RawEventsData", "RawTrackingData_Home_Team", "RawTrackingData_Away_Team"):
            name = "Sample_Game_{}_{}.csv".format(game, part)
            yield METRICA + "Sample_Game_{}/{}".format(game, name), os.path.join("metrica", name)
    for part in ("tracking.txt", "metadata.xml", "events.json"):
        name = "Sample_Game_3_" + part
        yield METRICA + "Sample_Game_3/" + name, os.path.join("metrica", name)
    for match in DFL_MATCHES:
        for prefix in ("DFL_02_01_matchinformation_", "DFL_04_03_positions_raw_observed_"):
            name = prefix + match + ".xml"
            yield DFL + name, os.path.join("dfl", name)


if __name__ == '__main__':
    for url, relative in files():
        path = os.path.join(EXTERNAL, relative)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            print("have", relative)
            continue
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        print("downloading", relative, flush=True)
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as error:   # report and continue: one failed file should not hide the rest
            print("  FAILED: {}".format(error))
            sys.exit(1)
