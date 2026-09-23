"""
09_augment_and_evaluate.py
--------------------------
THE ACTUAL EXPERIMENT.

Everything so far has been setup. This script answers the question the
whole project exists to ask:

    Does adding GAN-generated seizure windows to the training data
    improve the detector's ability to find real seizures?

--------------------------------------------------------------------
HOW THE COMPARISON IS KEPT HONEST
--------------------------------------------------------------------
1. PAIRED DESIGN. For every fold we train TWO detectors from scratch --
   one on real data only, one on real + synthetic -- and compare them on
   the SAME held-out recording. Same fold, same seed, same everything
   except the synthetic windows. A paired comparison removes fold-to-fold
   difficulty as a source of noise, which matters enormously when your
   baseline already varies by +/- 0.14 across folds.

2. SYNTHETIC DATA NEVER TOUCHES EVALUATION. Generated windows are added
   to the TRAINING set only. They never appear in the calibration
   recording or the held-out test recording. We give them a fake
   recording name, "SYNTHETIC", so they can never be selected as a fold.

3. WE REUSE SCRIPT 05'S CODE, NOT A COPY. The model, the training loop
   and the scoring functions are imported from 05_train_detector.py. If
   the two scripts drifted apart, the comparison would be meaningless.

4. THE VERDICT ACCOUNTS FOR NOISE. An average improvement smaller than
   the fold-to-fold spread is not evidence of anything. The script says
   so explicitly rather than letting you read a small positive number as
   a success.

--------------------------------------------------------------------
A NOTE ON WHAT A NULL RESULT MEANS
--------------------------------------------------------------------
If synthetic data does NOT help, that is a real finding, not a failure.
With 230 seizure windows from 7 seizures in ONE patient, a generator can
only recombine the variation that is already present. It cannot invent
seizure morphologies the patient never had. Reporting that honestly --
with a properly controlled baseline behind it -- is better work than
reporting an improvement you cannot defend.

Inputs:
    data/processed/chb01_windows.npz   (from 04_make_windows.py)
    models/eeg_cgan.pt                 (from 07_train_gan.py)

Outputs:
    results/augmentation_results.csv
    results/augmentation_comparison.png

Run from the project root:
    python scripts/09_augment_and_evaluate.py
    python scripts/09_augment_and_evaluate.py --n-synthetic 1000
    python scripts/09_augment_and_evaluate.py --epochs 4 --folds 2   # quick test
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the detector and the GAN exactly as they were defined, so this
# experiment cannot silently diverge from the baseline it compares against.
detector = importlib.import_module("05_train_detector")
gan = importlib.import_module("07_train_gan")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_NPZ = PROJECT_ROOT / "data" / "processed" / "chb01_windows.npz"
GAN_MODEL = PROJECT_ROOT / "models" / "eeg_cgan.pt"
RESULTS_CSV = PROJECT_ROOT / "results" / "augmentation_results.csv"
OUTPUT_PNG = PROJECT_ROOT / "results" / "augmentation_comparison.png"

N_SYNTHETIC = 500          # synthetic seizure windows added to each training set
RANDOM_SEED = 42

COLOR_BASE = "#2b6cb0"
COLOR_AUG = "#c0392b"


def generate_synthetic(n, device, seed=RANDOM_SEED):
    """Ask the trained generator for n synthetic SEIZURE windows."""
    ckpt = torch.load(GAN_MODEL, map_location=device, weights_only=False)

    G = gan.Generator(n_channels=ckpt["n_channels"],
                      latent_dim=ckpt["latent_dim"],
                      base=ckpt["base_channels"]).to(device)
    G.load_state_dict(ckpt["generator"])
    G.eval()

    torch.manual_seed(seed)
    out = []
    with torch.no_grad():
        # Generate in chunks so we never hold a huge tensor on the GPU.
        for i in range(0, n, 128):
            k = min(128, n - i)
            z = torch.randn(k, ckpt["latent_dim"], device=device)
            labels = torch.ones(k, dtype=torch.long, device=device)
            out.append(G(z, labels).cpu().numpy())

    return np.concatenate(out).astype(np.float16), ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-synthetic", type=int, default=N_SYNTHETIC,
                        help="synthetic seizure windows added per fold")
    parser.add_argument("--epochs", type=int, default=detector.EPOCHS)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--class-weight", choices=["balanced", "sqrt", "none"],
                        default="sqrt")
    parser.add_argument("--norm", choices=["group", "batch", "none"],
                        default="group")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = gan.pick_device(args.device)
    detector.DEVICE = device        # keep both modules on the same device

    print("=" * 70)
    print("09_augment_and_evaluate.py -- does synthetic EEG improve detection?")
    print("=" * 70)
    print(f"Device: {device}   epochs/fold: {args.epochs}   "
          f"synthetic windows: {args.n_synthetic}")

    for path, hint in [(DATA_NPZ, "04_make_windows.py"), (GAN_MODEL, "07_train_gan.py")]:
        if not path.exists():
            print(f"ERROR: {path} not found. Run scripts/{hint} first.")
            sys.exit(1)

    # --- real data --------------------------------------------------------
    data = np.load(DATA_NPZ, allow_pickle=False)
    X_real, y_real = data["X"], data["y"]
    metadata = data["metadata"]
    recordings_real = metadata["file"]
    stride_sec = float(data["stride_sec"])
    n_channels = X_real.shape[1]

    print(f"\nReal windows    : {len(y_real)}  ({int(y_real.sum())} seizure)")

    # --- synthetic data ----------------------------------------------------
    print(f"Generating {args.n_synthetic} synthetic seizure windows ...")
    synth, gan_ckpt = generate_synthetic(args.n_synthetic, device)
    print(f"  generator trained for {gan_ckpt['epochs']} epochs")
    print(f"  synthetic shape: {synth.shape}")

    # Stack real and synthetic into one array. Synthetic windows get the
    # recording name "SYNTHETIC" so they can never be picked as a test or
    # calibration fold -- only ever added to training.
    X_all = np.concatenate([X_real, synth], axis=0)
    y_all = np.concatenate([y_real, np.ones(len(synth), dtype=np.int64)])
    recordings_all = np.concatenate(
        [recordings_real, np.array(["SYNTHETIC"] * len(synth), dtype=recordings_real.dtype)]
    )
    synth_idx = np.arange(len(y_real), len(y_all))

    # --- folds -------------------------------------------------------------
    real_recordings = sorted(np.unique(recordings_real).tolist())
    seizure_recordings = sorted(
        {r for r in real_recordings if y_real[recordings_real == r].sum() > 0}
    )
    folds = seizure_recordings
    if args.folds is not None:
        folds = folds[: args.folds]

    print(f"\nFolds: {len(folds)} (leave-one-recording-out over seizure recordings)")
    print("Each fold trains TWO detectors: baseline and augmented.\n")

    rows = []
    start = time.time()

    for k, held_out in enumerate(folds, start=1):
        others = [r for r in seizure_recordings if r != held_out]
        calib_rec = others[k % len(others)]

        test_idx = np.flatnonzero(recordings_all == held_out)
        calib_idx = np.flatnonzero(recordings_all == calib_rec)

        # Baseline training set: real recordings only.
        train_base = np.flatnonzero(
            (recordings_all != held_out)
            & (recordings_all != calib_rec)
            & (recordings_all != "SYNTHETIC")
        )
        # Augmented training set: the same, plus the synthetic windows.
        train_aug = np.concatenate([train_base, synth_idx])

        hours = len(test_idx) * stride_sec / 3600.0

        print(f"  Fold {k}/{len(folds)}  test {held_out}, calibrate {calib_rec}")
        print(f"    baseline train {len(train_base)} | "
              f"augmented train {len(train_aug)} "
              f"(+{len(synth_idx)} synthetic seizure)")

        fold_scores = {}
        for tag, train_idx in [("baseline", train_base), ("augmented", train_aug)]:
            # Same seed for both arms, so the only difference is the data.
            _, y_true, y_prob, ca_true, ca_prob = detector.run_fold(
                X_all, y_all, train_idx, calib_idx, test_idx,
                args.epochs, n_channels,
                weight_mode=args.class_weight, norm=args.norm, quiet=True,
            )
            thr = detector.pick_threshold(ca_true, ca_prob)
            fold_scores[tag] = detector.score(y_true, y_prob, thr, hours)

        b, a = fold_scores["baseline"], fold_scores["augmented"]
        delta = a["auprc"] - b["auprc"]

        arrow = "+" if delta > 0 else ""
        print(f"    AUPRC  baseline {b['auprc']:.3f}  ->  augmented {a['auprc']:.3f}"
              f"   ({arrow}{delta:.3f})")
        print(f"    recall {b['recall']:.3f} -> {a['recall']:.3f} | "
              f"FA/h {b['fp_per_hour']:.1f} -> {a['fp_per_hour']:.1f}")

        rows.append({
            "fold": k, "held_out": held_out, "calibrated_on": calib_rec,
            "n_test_seizure": int(y_all[test_idx].sum()),
            "auprc_baseline": b["auprc"], "auprc_augmented": a["auprc"],
            "auprc_delta": delta,
            "recall_baseline": b["recall"], "recall_augmented": a["recall"],
            "fp_per_hour_baseline": b["fp_per_hour"],
            "fp_per_hour_augmented": a["fp_per_hour"],
        })

    df = pd.DataFrame(rows)

    # --- verdict ------------------------------------------------------------
    # Use the SAMPLE standard deviation (ddof=1) everywhere. With 7 folds the
    # difference from the population version is ~8%, and mixing the two would
    # make the noise band inconsistent with the table above it.
    def sd(a):
        a = np.asarray(a, dtype=float)
        return float(a.std(ddof=1)) if len(a) > 1 else float("nan")

    base_mean, base_std = df["auprc_baseline"].mean(), sd(df["auprc_baseline"])
    aug_mean, aug_std = df["auprc_augmented"].mean(), sd(df["auprc_augmented"])
    deltas = df["auprc_delta"].to_numpy()
    mean_delta, std_delta = float(deltas.mean()), sd(deltas)
    n_better = int((deltas > 0).sum())
    n_folds = len(df)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(df[["fold", "held_out", "auprc_baseline", "auprc_augmented",
              "auprc_delta"]].to_string(index=False,
                                        float_format=lambda v: f"{v:.3f}"))

    print(f"\n  baseline  AUPRC : {base_mean:.3f} +/- {base_std:.3f}")
    print(f"  augmented AUPRC : {aug_mean:.3f} +/- {aug_std:.3f}")
    print(f"\n  mean paired difference : {mean_delta:+.3f} +/- {std_delta:.3f}")
    print(f"  folds improved         : {n_better} of {n_folds}")

    print("\n" + "-" * 70)
    print("VERDICT")
    print("-" * 70)

    if n_folds < 5:
        print(f"  WARNING: only {n_folds} folds. This is a smoke test, not a result --")
        print("  run all folds before drawing any conclusion.\n")

    # An effect is only credible if it is (a) bigger than the fold-to-fold
    # noise AND (b) consistent across most folds. One big win in one fold
    # can drag the mean up while telling you nothing.
    consistent = n_better >= int(np.ceil(0.7 * n_folds))
    exceeds_noise = abs(mean_delta) > std_delta

    if not exceeds_noise:
        print("  NO DETECTABLE EFFECT.")
        print(f"  The mean change ({mean_delta:+.3f}) is smaller than the variation")
        print(f"  between folds ({std_delta:.3f}), so it cannot be distinguished")
        print("  from noise. This does NOT prove synthetic data is useless -- it")
        print("  proves that with 7 seizures from one patient, any real effect is")
        print("  too small for this experiment to resolve.")
    elif mean_delta > 0 and consistent:
        print("  SYNTHETIC DATA HELPED.")
        print(f"  Mean AUPRC improved by {mean_delta:+.3f}, larger than the")
        print(f"  fold-to-fold spread ({std_delta:.3f}), and {n_better}/{n_folds} folds")
        print("  improved individually. That consistency is what makes it credible.")
    elif mean_delta > 0:
        print("  POSSIBLE IMPROVEMENT, NOT YET CONVINCING.")
        print(f"  Mean AUPRC rose {mean_delta:+.3f}, but only {n_better}/{n_folds} folds")
        print("  improved. A real effect should show up in most folds, not a few.")
        print("  A single large win in one fold can carry the mean on its own.")
    else:
        print("  SYNTHETIC DATA HURT.")
        print(f"  Mean AUPRC fell by {mean_delta:.3f}. The most likely reason is that")
        print("  the synthetic windows carry structure real seizures do not, so the")
        print("  detector learns to spot the generator instead of learning seizures.")

    print("\n  Reminder: this compares against YOUR baseline, on YOUR folds.")
    print("  Report the paired difference, not the two means separately.")

    # --- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(df))
    w = 0.38
    axes[0].bar(x - w / 2, df["auprc_baseline"], w, label="real data only",
                color=COLOR_BASE)
    axes[0].bar(x + w / 2, df["auprc_augmented"], w,
                label=f"+ {args.n_synthetic} synthetic", color=COLOR_AUG)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([r.replace("chb01_", "").replace(".edf", "")
                             for r in df["held_out"]])
    axes[0].set_xlabel("held-out recording")
    axes[0].set_ylabel("AUPRC")
    axes[0].set_title("Per-fold detection quality", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.5)

    colours = [COLOR_AUG if d > 0 else COLOR_BASE for d in deltas]
    axes[1].bar(x, deltas, 0.6, color=colours)
    axes[1].axhline(0, color="black", linewidth=1)
    # The noise band: changes inside this range are not distinguishable.
    axes[1].axhspan(-std_delta, std_delta, color="grey", alpha=0.18,
                    label="within fold-to-fold noise")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r.replace("chb01_", "").replace(".edf", "")
                             for r in df["held_out"]])
    axes[1].set_xlabel("held-out recording")
    axes[1].set_ylabel("change in AUPRC")
    axes[1].set_title(f"Effect of augmentation\nmean {mean_delta:+.3f} "
                      f"+/- {std_delta:.3f}", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.5)

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle("NeuroSynth -- does synthetic seizure EEG improve the detector?",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=130)
    plt.close(fig)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)

    print(f"\n  Ran in {(time.time() - start) / 60:.1f} minutes.")
    print(f"  Saved table  : {RESULTS_CSV}")
    print(f"  Saved figure : {OUTPUT_PNG}")
    print("\nDone.")


if __name__ == "__main__":
    main()
