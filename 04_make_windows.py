"""
04_make_windows.py
------------------
Turn raw EDF recordings into a labelled dataset of fixed-length EEG windows.

Pipeline for each EDF file:
    1. load the file with MNE
    2. pick a stable subset of 18 bipolar channels
    3. band-pass filter 0.5 - 40 Hz
    4. resample to 256 Hz (if needed)
    5. cut into 4-second windows with a 2-second stride
    6. label each window 1 (seizure) if it overlaps any seizure interval,
       otherwise 0 (non-seizure)

Inputs:
    data/raw/chbmit/chb01/chb01_01.edf
    data/raw/chbmit/chb01/chb01_03.edf
    data/processed/chb01_seizure_intervals.csv   (made by 03_parse_summary.py)

Output:
    data/processed/chb01_windows_tiny.npz

The .npz contains:
    X            float32 [num_windows, channels, time_points]  -- the EEG, in microvolts
    y            int64   [num_windows]                         -- 0 = non-seizure, 1 = seizure
    metadata     structured array with fields (file, start_sec, end_sec)
    channels     the channel names, in the same order as axis 1 of X
    sfreq        sampling rate in Hz (256)
    window_sec   window length in seconds (4)
    stride_sec   hop between window starts in seconds (2)

Run from the project root:
    python scripts/04_make_windows.py
"""

import re
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Configuration -- all the knobs in one place
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "chbmit" / "chb01"
INTERVALS_CSV = PROJECT_ROOT / "data" / "processed" / "chb01_seizure_intervals.csv"
OUTPUT_NPZ = PROJECT_ROOT / "data" / "processed" / "chb01_windows.npz"

# Which EDF files to process. Leave as None to automatically use every
# .edf file sitting in data/raw/chbmit/chb01/ -- so downloading more
# recordings is all you need to do to grow the dataset.
EDF_FILES = None

# Windowing settings
WINDOW_SEC = 4.0        # each window is 4 seconds long
STRIDE_SEC = 2.0        # start a new window every 2 seconds (so windows overlap by 50%)
TARGET_SFREQ = 256.0    # resample everything to 256 Hz

# Filtering settings (band-pass)
L_FREQ = 0.5            # remove very slow drift below 0.5 Hz
H_FREQ = 40.0           # remove muscle artefacts / line noise above 40 Hz

# 4 seconds * 256 Hz = 1024 time points per window
TIME_POINTS = int(round(WINDOW_SEC * TARGET_SFREQ))

# The 18 bipolar channels that are present in (almost) every CHB-MIT record.
# Using a fixed list means every file produces the same channel ordering,
# which is essential for feeding a neural network later.
WANTED_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FZ-CZ", "CZ-PZ",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
]


# ---------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------

def normalise(name: str) -> str:
    """
    Make channel-name comparison robust.

    CHB-MIT EDF files spell channels slightly differently across records
    ('FP1-F7' vs 'Fp1-F7', stray spaces, etc.). We also have to deal with
    one specific quirk: chb01 EDF files list the label 'T8-P8' TWICE, and
    MNE de-duplicates them into 'T8-P8-0' and 'T8-P8-1'. Stripping a
    trailing '-<digits>' turns 'T8-P8-0' back into 'T8-P8'.

    Note this is safe for names like 'FT9-FT10' because the part after the
    last dash there ('FT10') is not purely digits.
    """
    clean = name.strip().upper().replace(" ", "")
    clean = re.sub(r"-\d+$", "", clean)     # 'T8-P8-0' -> 'T8-P8'
    return clean


def pick_channels(raw: mne.io.BaseRaw):
    """
    Find our 18 wanted channels inside this recording.

    Returns
    -------
    picked_names : list[str]
        The ACTUAL channel names in the file, ordered to match WANTED_CHANNELS.
    missing : list[str]
        Any wanted channels that this file simply does not have.
    """
    # Map normalised name -> the first raw channel name that matches it.
    lookup = {}
    for actual_name in raw.ch_names:
        key = normalise(actual_name)
        if key not in lookup:           # keep the FIRST occurrence of a duplicate
            lookup[key] = actual_name

    picked_names = []
    missing = []
    for wanted in WANTED_CHANNELS:
        key = normalise(wanted)
        if key in lookup:
            picked_names.append(lookup[key])
        else:
            missing.append(wanted)

    return picked_names, missing


def load_seizure_intervals(csv_path: Path) -> dict:
    """
    Read the CSV written by 03_parse_summary.py and return a dictionary:

        {"chb01_01.edf": [], "chb01_03.edf": [(2996.0, 3036.0)]}

    Files with no seizures map to an empty list.
    """
    df = pd.read_csv(csv_path)

    intervals = {name: [] for name in df["file"].unique()}

    # Only rows with has_seizure == 1 carry real start/end times.
    for _, row in df[df["has_seizure"] == 1].iterrows():
        intervals[row["file"]].append((float(row["start_sec"]), float(row["end_sec"])))

    return intervals


def overlaps_any(win_start: float, win_end: float, intervals) -> bool:
    """
    Does the window [win_start, win_end) touch any seizure interval [s, e)?

    Two intervals overlap when each one starts before the other ends.
    Even a single overlapping sample counts, which is what we want:
    a window that contains part of a seizure IS a seizure window.
    """
    for (s, e) in intervals:
        if win_start < e and win_end > s:
            return True
    return False


# ---------------------------------------------------------------------
# 3. Process one EDF file
# ---------------------------------------------------------------------

def process_file(edf_path: Path, seizure_intervals):
    """
    Load, filter, resample and window a single EDF file.

    Returns (X, y, meta_rows, picked_channel_names) where
        X          float32 [n_windows, n_channels, TIME_POINTS]
        y          int64   [n_windows]
        meta_rows  list of (file_name, start_sec, end_sec) tuples
    """
    print(f"\n--- {edf_path.name} ---")

    # preload=True pulls the whole recording into RAM so we can filter it.
    # A one-hour CHB-MIT file is roughly 150 MB as floats -- fine on a laptop.
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    print(f"  channels in file : {len(raw.ch_names)}")
    print(f"  sampling rate    : {raw.info['sfreq']:.1f} Hz")
    print(f"  duration         : {raw.n_times / raw.info['sfreq']:.1f} s")

    # --- step 1: keep only our 18 bipolar channels, in a fixed order ----
    picked_names, missing = pick_channels(raw)
    if missing:
        # We stop rather than silently training on a different channel set,
        # because inconsistent channels across files would corrupt the dataset.
        print(f"  ERROR: these wanted channels are missing: {missing}")
        raise ValueError(f"{edf_path.name} is missing channels: {missing}")

    raw.pick(picked_names)
    # pick() does not guarantee ordering in every MNE version, so reorder explicitly.
    raw.reorder_channels(picked_names)
    print(f"  kept {len(picked_names)} channels")

    # Show any place where the file's label differs from our canonical name,
    # so you can see exactly what got matched to what.
    renames = [(a, w) for a, w in zip(picked_names, WANTED_CHANNELS) if a != w]
    if renames:
        print(f"  matched (file label -> canonical): {renames}")

    # --- step 2: band-pass filter 0.5 - 40 Hz --------------------------
    # This removes slow electrode drift and high-frequency muscle noise,
    # keeping the frequency band where seizure activity lives.
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, fir_design="firwin", verbose="ERROR")
    print(f"  band-pass filtered {L_FREQ}-{H_FREQ} Hz")

    # --- step 3: resample to a common sampling rate --------------------
    # CHB-MIT is already 256 Hz, but doing this explicitly means the script
    # still works if you later add a record sampled at a different rate.
    if abs(raw.info["sfreq"] - TARGET_SFREQ) > 1e-6:
        raw.resample(TARGET_SFREQ, verbose="ERROR")
        print(f"  resampled to {TARGET_SFREQ} Hz")
    else:
        print(f"  already at {TARGET_SFREQ} Hz, no resampling needed")

    # --- step 4: get the data as a plain NumPy array --------------------
    # MNE stores EEG in volts; multiply by 1e6 to get microvolts, which are
    # the units neurologists actually use and are nicer numbers for a network.
    data = raw.get_data() * 1e6          # shape: [n_channels, n_samples]
    n_channels, n_samples = data.shape

    # --- step 5: slice into overlapping windows -------------------------
    stride_samples = int(round(STRIDE_SEC * TARGET_SFREQ))

    windows = []
    labels = []
    meta_rows = []

    start_sample = 0
    while start_sample + TIME_POINTS <= n_samples:
        stop_sample = start_sample + TIME_POINTS

        # Convert sample indices to seconds so we can compare with the
        # seizure annotations, which are given in seconds.
        win_start_sec = start_sample / TARGET_SFREQ
        win_end_sec = stop_sample / TARGET_SFREQ

        windows.append(data[:, start_sample:stop_sample])
        labels.append(1 if overlaps_any(win_start_sec, win_end_sec, seizure_intervals) else 0)
        meta_rows.append((edf_path.name, win_start_sec, win_end_sec))

        start_sample += stride_samples

    # float16 halves the file size and RAM cost. EEG is recorded by the
    # amplifier at 16-bit resolution anyway, so we are not throwing away
    # anything the hardware actually measured. The training script casts
    # back to float32 before it touches the network.
    X = np.stack(windows).astype(np.float16)     # [n_windows, n_channels, TIME_POINTS]
    y = np.array(labels, dtype=np.int64)

    print(f"  windows made     : {len(y)}  (shape {X.shape})")
    print(f"  seizure windows  : {int(y.sum())}")
    print(f"  non-seizure      : {int((y == 0).sum())}")

    return X, y, meta_rows, picked_names


# ---------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------

def main():
    print("=" * 60)
    print("04_make_windows.py -- building labelled EEG windows")
    print("=" * 60)
    print(f"window = {WINDOW_SEC}s, stride = {STRIDE_SEC}s, "
          f"target rate = {TARGET_SFREQ} Hz -> {TIME_POINTS} time points per window")

    # --- check inputs exist -------------------------------------------
    if not INTERVALS_CSV.exists():
        print(f"ERROR: {INTERVALS_CSV} not found.")
        print("Run scripts/03_parse_summary.py first.")
        sys.exit(1)

    seizure_map = load_seizure_intervals(INTERVALS_CSV)
    print(f"\nSeizure intervals loaded from {INTERVALS_CSV.name}:")
    for name, iv in seizure_map.items():
        print(f"  {name}: {iv if iv else 'none'}")

    # --- decide which EDF files to process -----------------------------
    if EDF_FILES is None:
        edf_names = sorted(p.name for p in RAW_DIR.glob("*.edf"))
    else:
        edf_names = list(EDF_FILES)

    if not edf_names:
        print(f"\nERROR: no .edf files in {RAW_DIR}. Run scripts/02_download_chbmit.py first.")
        sys.exit(1)

    print(f"\nProcessing {len(edf_names)} recordings: {edf_names}")

    # --- process every EDF file ---------------------------------------
    all_X, all_y, all_meta = [], [], []
    channel_names = None

    for name in edf_names:
        edf_path = RAW_DIR / name
        if not edf_path.exists():
            print(f"\nERROR: {edf_path} not found. Run scripts/02_download_chbmit.py first.")
            sys.exit(1)

        X, y, meta_rows, picked = process_file(edf_path, seizure_map.get(name, []))

        # Every file must end up with the SAME channel ordering.
        if channel_names is None:
            channel_names = picked
        elif [normalise(c) for c in picked] != [normalise(c) for c in channel_names]:
            print("ERROR: channel ordering differs between files. Aborting.")
            sys.exit(1)

        all_X.append(X)
        all_y.append(y)
        all_meta.extend(meta_rows)

    # --- stack everything into one dataset -----------------------------
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Store metadata as a structured array: this saves into .npz cleanly
    # without needing pickle, and you can index it like meta["file"].
    meta_dtype = np.dtype([("file", "U32"), ("start_sec", "f8"), ("end_sec", "f8")])
    metadata = np.array(all_meta, dtype=meta_dtype)

    # --- summary --------------------------------------------------------
    n_seizure = int(y.sum())
    n_total = int(y.size)
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  X shape        : {X.shape}   (windows, channels, time points)")
    print(f"  y shape        : {y.shape}")
    print(f"  total windows  : {n_total}")
    print(f"  seizure (y=1)  : {n_seizure}  ({100.0 * n_seizure / n_total:.2f}%)")
    print(f"  non-seizure    : {n_total - n_seizure}")
    # We store the CANONICAL names (WANTED_CHANNELS) rather than whatever the
    # EDF happened to call them, so downstream scripts always see 'T8-P8'
    # instead of MNE's de-duplicated 'T8-P8-0'. The data order is identical.
    channel_names = list(WANTED_CHANNELS)
    print(f"  channels ({len(channel_names)}) : {channel_names}")
    print(f"  amplitude range: {X.min():.1f} to {X.max():.1f} microvolts")
    print(f"  dtype          : {X.dtype}")

    # Per-recording breakdown -- this is what the recording-level split in
    # script 05 works from, so it is worth eyeballing.
    print("\n  per recording:")
    for name in edf_names:
        mask = metadata["file"] == name
        n_win = int(mask.sum())
        n_pos = int(y[mask].sum())
        print(f"    {name:<18} {n_win:>6} windows, {n_pos:>4} seizure")

    if n_seizure == 0:
        print("\nWARNING: no seizure windows found. Check the CSV from script 03.")

    # --- save ------------------------------------------------------------
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        X=X,
        y=y,
        metadata=metadata,
        channels=np.array(channel_names),
        sfreq=np.float64(TARGET_SFREQ),
        window_sec=np.float64(WINDOW_SEC),
        stride_sec=np.float64(STRIDE_SEC),
    )
    size_mb = OUTPUT_NPZ.stat().st_size / 1e6
    print(f"\nSaved dataset to: {OUTPUT_NPZ}  ({size_mb:.1f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()