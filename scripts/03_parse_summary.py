"""
03_parse_summary.py
-------------------
Read the CHB-MIT summary text file for patient chb01 and turn the
seizure annotations into a tidy CSV table.

Input:
    data/raw/chbmit/chb01/chb01-summary.txt

Output:
    data/processed/chb01_seizure_intervals.csv

Columns:
    file, seizure_index, start_sec, end_sec, has_seizure

Why we need this:
    The summary file is human-readable text, not a table. Later scripts
    (04_make_windows.py) need machine-readable seizure start/end times
    so they can label each EEG window as seizure / non-seizure.

Run from the project root:
    python scripts/03_parse_summary.py
"""

import re
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

# All paths are relative to the project root (the folder that contains
# data/, scripts/, models/ ...). We compute the project root from this
# file's location so the script works no matter where you launch it from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUMMARY_PATH = PROJECT_ROOT / "data" / "raw" / "chbmit" / "chb01" / "chb01-summary.txt"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "chb01_seizure_intervals.csv"

# The summary file describes all 42 chb01 recordings, but we have only
# downloaded some of them. We keep a row for every EDF that is actually
# present in data/raw/chbmit/chb01/ -- so this script automatically adapts
# when you download more files later. Nothing to edit by hand.
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "chbmit" / "chb01"


# ---------------------------------------------------------------------
# 2. Regular expressions that match the lines we care about
# ---------------------------------------------------------------------
#
# A typical block inside chb01-summary.txt looks like this:
#
#     File Name: chb01_03.edf
#     File Start Time: 13:43:04
#     File End Time: 14:43:04
#     Number of Seizures in File: 1
#     Seizure Start Time: 2996 seconds
#     Seizure End Time: 3036 seconds
#
# When a file contains more than one seizure, the lines are numbered:
#
#     Number of Seizures in File: 2
#     Seizure 1 Start Time: 327 seconds
#     Seizure 1 End Time: 420 seconds
#     Seizure 2 Start Time: 1862 seconds
#     Seizure 2 End Time: 1963 seconds
#
# The regexes below handle BOTH forms (with and without the number).

RE_FILE_NAME = re.compile(r"^File Name:\s*(?P<name>\S+)", re.IGNORECASE)

RE_NUM_SEIZURES = re.compile(
    r"^Number of Seizures in File:\s*(?P<count>\d+)", re.IGNORECASE
)

# "Seizure Start Time: 2996 seconds"  or  "Seizure 1 Start Time: 327 seconds"
RE_SEIZURE_START = re.compile(
    r"^Seizure\s*(?P<index>\d+)?\s*Start Time:\s*(?P<seconds>[\d.]+)\s*seconds",
    re.IGNORECASE,
)

RE_SEIZURE_END = re.compile(
    r"^Seizure\s*(?P<index>\d+)?\s*End Time:\s*(?P<seconds>[\d.]+)\s*seconds",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------
# 3. The parser
# ---------------------------------------------------------------------

def parse_summary(summary_path: Path) -> pd.DataFrame:
    """
    Walk through the summary file line by line and collect, for every
    EDF file mentioned, its list of seizure intervals.

    Returns a DataFrame with one row per (file, seizure). Files without
    seizures still get exactly one row, with empty start/end times.
    """
    text = summary_path.read_text(encoding="utf-8", errors="ignore")

    rows = []

    # State variables: which file are we currently reading about, and
    # what seizure intervals have we seen for it so far?
    current_file = None
    current_starts = []
    current_ends = []

    def flush_current_file():
        """Turn whatever we collected for `current_file` into table rows."""
        if current_file is None:
            return

        if len(current_starts) == 0:
            # No seizures in this file -> a single row with has_seizure = 0
            rows.append(
                {
                    "file": current_file,
                    "seizure_index": 0,
                    "start_sec": "",
                    "end_sec": "",
                    "has_seizure": 0,
                }
            )
        else:
            # One row per seizure interval, numbered starting at 1
            for i, (start, end) in enumerate(zip(current_starts, current_ends), start=1):
                rows.append(
                    {
                        "file": current_file,
                        "seizure_index": i,
                        "start_sec": start,
                        "end_sec": end,
                        "has_seizure": 1,
                    }
                )

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # --- a new "File Name:" line starts a new block ---------------
        match = RE_FILE_NAME.match(line)
        if match:
            # First save whatever we collected for the previous file.
            flush_current_file()

            current_file = match.group("name")
            current_starts = []
            current_ends = []
            continue

        # We ignore everything before the first "File Name:" line
        # (the header that lists channels, sampling rate, etc.).
        if current_file is None:
            continue

        # --- seizure start / end lines -------------------------------
        # NOTE: check "End Time" BEFORE "Start Time" is not required
        # because the two patterns are mutually exclusive, but we do
        # check start first simply for readability.
        match = RE_SEIZURE_START.match(line)
        if match:
            current_starts.append(float(match.group("seconds")))
            continue

        match = RE_SEIZURE_END.match(line)
        if match:
            current_ends.append(float(match.group("seconds")))
            continue

        # "Number of Seizures in File: N" is useful as a sanity check
        # later, but we do not strictly need it because we count the
        # start/end lines directly.
        match = RE_NUM_SEIZURES.match(line)
        if match:
            continue

    # Do not forget the very last file in the document.
    flush_current_file()

    return pd.DataFrame(rows, columns=["file", "seizure_index", "start_sec", "end_sec", "has_seizure"])


# ---------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------

def main():
    print("=" * 60)
    print("03_parse_summary.py -- extracting seizure intervals")
    print("=" * 60)

    # --- check the input file exists ---------------------------------
    if not SUMMARY_PATH.exists():
        print(f"ERROR: summary file not found at:\n  {SUMMARY_PATH}")
        print("Did you run scripts/02_download_chbmit.py first?")
        sys.exit(1)

    print(f"Reading summary: {SUMMARY_PATH}")

    # --- parse everything --------------------------------------------
    all_files_df = parse_summary(SUMMARY_PATH)
    print(f"Found {all_files_df['file'].nunique()} EDF files mentioned in the summary.")

    # --- see which EDF files we actually downloaded --------------------
    present = sorted(p.name for p in RAW_DIR.glob("*.edf"))
    if not present:
        print(f"ERROR: no .edf files found in {RAW_DIR}")
        print("Run scripts/02_download_chbmit.py first.")
        sys.exit(1)

    print(f"EDF files on disk: {len(present)}")

    df = all_files_df[all_files_df["file"].isin(present)].copy()

    # If an EDF exists on disk but the summary never mentioned it, still
    # add a "no seizure" row so script 04 does not silently skip the file.
    for name in present:
        if name not in set(df["file"]):
            print(f"WARNING: {name} was not found in the summary; assuming no seizures.")
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [{
                            "file": name,
                            "seizure_index": 0,
                            "start_sec": "",
                            "end_sec": "",
                            "has_seizure": 0,
                        }]
                    ),
                ],
                ignore_index=True,
            )

    # Sort for a stable, readable CSV.
    df = df.sort_values(["file", "seizure_index"]).reset_index(drop=True)

    # --- save ---------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # --- report -------------------------------------------------------
    print()
    print("Resulting table:")
    print(df.to_string(index=False))
    print()

    total_seizures = 0
    total_ictal_sec = 0.0

    for name in present:
        sub = df[df["file"] == name]
        n_seizures = int(sub["has_seizure"].sum())
        if n_seizures == 0:
            print(f"  {name}: no seizures")
        else:
            for _, row in sub[sub["has_seizure"] == 1].iterrows():
                duration = float(row["end_sec"]) - float(row["start_sec"])
                total_seizures += 1
                total_ictal_sec += duration
                print(
                    f"  {name}: seizure {int(row['seizure_index'])} "
                    f"from {row['start_sec']}s to {row['end_sec']}s "
                    f"({duration:.0f}s long)"
                )

    print()
    print(f"Recordings         : {len(present)}")
    print(f"Recordings with a seizure : {df[df['has_seizure'] == 1]['file'].nunique()}")
    print(f"Total seizures     : {total_seizures}")
    print(f"Total seizure time : {total_ictal_sec:.0f} s")
    print()
    print(f"Saved CSV to: {OUTPUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()