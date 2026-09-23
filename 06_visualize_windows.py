"""
06_visualize_windows.py
-----------------------
Sanity-check the dataset with your eyes.

Plots one non-seizure window and one seizure window side by side, drawn
the way clinical EEG is normally displayed: every channel on its own
horizontal line, stacked vertically, sharing one time axis.

Input:
    data/processed/chb01_windows.npz   (made by 04_make_windows.py)

Output:
    results/example_eeg_windows.png

Why bother:
    If your labels are wrong, or your filter destroyed the signal, or the
    channels got scrambled, you will usually SEE it here long before the
    training metrics tell you. Always look at your data.

Run from the project root:
    python scripts/06_visualize_windows.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # render to a file, no interactive window needed
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_NPZ = PROJECT_ROOT / "data" / "processed" / "chb01_windows.npz"
OUTPUT_PNG = PROJECT_ROOT / "results" / "example_eeg_windows.png"

# Vertical distance between stacked channel traces, in microvolts.
# Leave as None to size it automatically from the data (recommended --
# seizure windows are much larger in amplitude than background EEG, and a
# fixed spacing makes them collide). Set a number to override.
CHANNEL_SPACING_UV = None

# Colours chosen to stay distinguishable in greyscale and for colour-blind readers.
COLOR_NON_SEIZURE = "#2b6cb0"   # blue
COLOR_SEIZURE = "#c0392b"       # red


# ---------------------------------------------------------------------
# 2. Drawing helper
# ---------------------------------------------------------------------

def choose_spacing(windows):
    """
    Pick a vertical gap between traces so that neighbouring channels do not
    collide. We use the 99th percentile of |amplitude| (robust to a single
    spike) across every window we are about to draw, times a small factor.
    Both panels must share one spacing because they share a y-axis.
    """
    if CHANNEL_SPACING_UV is not None:
        return float(CHANNEL_SPACING_UV)

    p99 = max(float(np.percentile(np.abs(w.astype(np.float32)), 99)) for w in windows)
    return max(60.0, 2.2 * p99)     # never smaller than 60 uV


def plot_window(ax, window, channel_names, times, colour, title, spacing):
    """
    Draw one EEG window on a matplotlib axis.

    Parameters
    ----------
    window  : array [n_channels, n_time_points], microvolts
    times   : array [n_time_points], seconds from the start of the window
    spacing : vertical gap between channel traces, microvolts
    """
    n_channels = window.shape[0]

    for i in range(n_channels):
        # Channel 0 is drawn at the TOP, so we offset downwards.
        offset = (n_channels - 1 - i) * spacing
        ax.plot(times, window[i].astype(np.float32) + offset, color=colour, linewidth=0.7)

    # Label the y-axis with channel names instead of numbers.
    tick_positions = [(n_channels - 1 - i) * spacing for i in range(n_channels)]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(channel_names, fontsize=7)

    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-spacing, n_channels * spacing)
    ax.set_xlabel("time within window (seconds)")
    ax.set_title(title, fontsize=10)

    # Light vertical gridlines help you judge frequency by eye.
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# ---------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------

def main():
    print("=" * 60)
    print("06_visualize_windows.py -- plotting example EEG windows")
    print("=" * 60)

    if not DATA_NPZ.exists():
        print(f"ERROR: {DATA_NPZ} not found. Run scripts/04_make_windows.py first.")
        sys.exit(1)

    data = np.load(DATA_NPZ, allow_pickle=False)
    # stored as float16 to keep the file small; cast up for plotting
    X = data["X"]                       # [n_windows, channels, time_points]
    y = data["y"]
    metadata = data["metadata"]         # fields: file, start_sec, end_sec
    channel_names = [str(c) for c in data["channels"]]
    sfreq = float(data["sfreq"])

    # --- find one example of each class --------------------------------
    seizure_idx = np.flatnonzero(y == 1)
    non_seizure_idx = np.flatnonzero(y == 0)

    if len(seizure_idx) == 0:
        print("ERROR: no seizure windows in the dataset -- nothing to compare.")
        sys.exit(1)

    # Pick the MIDDLE seizure window: it sits deepest inside the seizure,
    # so it shows the clearest ictal pattern. The first and last seizure
    # windows only partly overlap the event.
    pick_seizure = int(seizure_idx[len(seizure_idx) // 2])

    # Pick a non-seizure window from the same recording, far away in time,
    # so the comparison is not confounded by a different file.
    seizure_file = metadata["file"][pick_seizure]
    same_file_clean = [
        i for i in non_seizure_idx
        if metadata["file"][i] == seizure_file
        and abs(metadata["start_sec"][i] - metadata["start_sec"][pick_seizure]) > 300
    ]
    pick_clean = int(same_file_clean[len(same_file_clean) // 2]) if same_file_clean \
        else int(non_seizure_idx[len(non_seizure_idx) // 2])

    print(f"\nNon-seizure example: window #{pick_clean} "
          f"({metadata['file'][pick_clean]}, "
          f"{metadata['start_sec'][pick_clean]:.0f}-{metadata['end_sec'][pick_clean]:.0f}s)")
    print(f"Seizure example    : window #{pick_seizure} "
          f"({metadata['file'][pick_seizure]}, "
          f"{metadata['start_sec'][pick_seizure]:.0f}-{metadata['end_sec'][pick_seizure]:.0f}s)")

    # --- build the figure ------------------------------------------------
    times = np.arange(X.shape[2]) / sfreq       # 0 .. 4 seconds

    spacing = choose_spacing([X[pick_clean], X[pick_seizure]])
    print(f"Trace spacing      : {spacing:.0f} uV")

    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)

    plot_window(
        axes[0], X[pick_clean], channel_names, times, COLOR_NON_SEIZURE,
        f"NON-SEIZURE  (label 0)\n{metadata['file'][pick_clean]}  "
        f"t = {metadata['start_sec'][pick_clean]:.0f}-{metadata['end_sec'][pick_clean]:.0f}s",
        spacing,
    )
    plot_window(
        axes[1], X[pick_seizure], channel_names, times, COLOR_SEIZURE,
        f"SEIZURE  (label 1)\n{metadata['file'][pick_seizure]}  "
        f"t = {metadata['start_sec'][pick_seizure]:.0f}-{metadata['end_sec'][pick_seizure]:.0f}s",
        spacing,
    )

    # A scale bar tells the reader how big the wiggles actually are.
    bar_bottom = -spacing * 0.8
    axes[0].plot([0.15, 0.15], [bar_bottom, bar_bottom + 100],
                 color="black", linewidth=2)
    axes[0].text(0.22, bar_bottom + 50, "100 uV", fontsize=8, va="center")

    fig.suptitle(
        "NeuroSynth -- example 4-second EEG windows (18 bipolar channels, 0.5-40 Hz, 256 Hz)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # --- save --------------------------------------------------------------
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)

    print(f"\nSaved figure to: {OUTPUT_PNG}")
    print("Done.")


if __name__ == "__main__":
    main()