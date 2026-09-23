"""
05_train_detector.py
--------------------
Train and HONESTLY evaluate a baseline 1D CNN seizure detector.

This replaces the earlier random-split version, which had two problems
that made its scores meaningless:

  1. Windows overlap by 50%, so a random split put nearly-identical
     windows in both train and validation. The model was recognising
     samples it had effectively already seen.

  2. It picked the best epoch by validation F1 and then reported that
     same validation set as the final score. Selecting on your test set
     and then reporting it always looks good, because you are taking the
     maximum of a noisy sequence.

Both are fixed here:

  1. LEAVE-ONE-RECORDING-OUT cross-validation. Each fold holds out one
     entire recording. No window from the held-out hour is ever seen in
     training, in any form, so there is no leakage.

  2. FIXED number of epochs, no early stopping, no best-epoch selection.
     We report the final epoch. Nothing is tuned on the held-out data.

Because there are 7 recordings with seizures, we get 7 folds. Reporting
mean +/- standard deviation across folds tells you not just how well the
detector does, but how much that number moves around -- which is the
thing you need to know before claiming a GAN improved it.

Input:
    data/processed/chb01_windows.npz   (made by 04_make_windows.py)

Output:
    models/baseline_eeg_cnn.pt         (trained on ALL recordings)
    results/baseline_cv_results.csv    (per-fold metrics)

Run from the project root:
    python scripts/05_train_detector.py

Optional flags, handy while you are experimenting:
    python scripts/05_train_detector.py --epochs 5 --folds 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_NPZ = PROJECT_ROOT / "data" / "processed" / "chb01_windows.npz"
MODEL_OUT = PROJECT_ROOT / "models" / "baseline_eeg_cnn.pt"
RESULTS_CSV = PROJECT_ROOT / "results" / "baseline_cv_results.csv"

BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 2. Dataset class
# ---------------------------------------------------------------------

class EEGWindowDataset(Dataset):
    """
    Serves EEG windows to PyTorch.

    We keep ONE copy of the full X array and hand this class a list of
    indices, rather than slicing X into per-fold copies. With 14k windows
    that slicing would waste hundreds of megabytes per fold.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.X = X                      # float16, shared across folds
        self.y = y
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        # Cast float16 -> float32 here. We store float16 on disk to keep
        # the file small, but the network needs float32 to train stably.
        window = torch.from_numpy(self.X[idx].astype(np.float32))

        # --- per-window, per-channel z-scoring ------------------------
        # Standardising each channel of each window makes the network
        # focus on the SHAPE of the signal, not its absolute amplitude,
        # which drifts between electrodes and between recordings.
        mean = window.mean(dim=1, keepdim=True)
        std = window.std(dim=1, keepdim=True)
        window = (window - mean) / (std + 1e-6)

        return window, int(self.y[idx])


# ---------------------------------------------------------------------
# 3. The model -- same small 1D CNN as before
# ---------------------------------------------------------------------

class BaselineEEGCNN(nn.Module):
    """
    Three convolution blocks -> global average pool -> linear classifier.

    Shape walkthrough for a batch of 64:
        input                      [64, 18, 1024]
        block 1 (conv + pool /4)   [64, 16,  256]
        block 2 (conv + pool /4)   [64, 32,   64]
        block 3 (conv + pool /4)   [64, 64,   16]
        global average over time   [64, 64]
        linear                     [64, 2]
    """

    def __init__(self, n_channels: int = 18, n_classes: int = 2, norm: str = "group"):
        super().__init__()

        # NORMALISATION CHOICE -- this matters more than it looks.
        #
        # BatchNorm stores running mean/variance collected from the TRAINING
        # recordings and reuses them at evaluation time. When the held-out
        # recording has even slightly different statistics, every output
        # score shifts up or down together. In testing that made the
        # decision threshold untransferable between recordings: false alarms
        # ranged from 0 to ~1770 per hour across folds even though ranking
        # quality (AUPRC) stayed at 0.97 +/- 0.03.
        #
        # GroupNorm normalises each sample using only that sample's own
        # statistics. Nothing carries over from the training set, so scores
        # do not drift when the recording changes. Default here for that
        # reason; use --norm batch to reproduce the old behaviour.
        def make_norm(out_ch):
            if norm == "batch":
                return nn.BatchNorm1d(out_ch)
            elif norm == "group":
                return nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
            elif norm == "none":
                return nn.Identity()
            raise ValueError(f"unknown norm: {norm}")

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
                make_norm(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=4),
            )

        self.features = nn.Sequential(
            block(n_channels, 16),
            block(16, 32),
            block(32, 64),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.classifier(x)


# ---------------------------------------------------------------------
# 4. Training / evaluation helpers
# ---------------------------------------------------------------------

def make_class_weights(y_train: np.ndarray, mode: str = "sqrt") -> torch.Tensor:
    """
    Weight the rare seizure class more heavily in the loss.

    Without any weighting, a model that always says "non-seizure" scores
    ~98% accuracy and is useless. But full inverse-frequency weighting
    over-corrects at this imbalance: the seizure class gets a weight near
    85, and the model learns to flag most of the recording, giving recall
    1.00 with precision 0.015. That is equally useless, and worse for us --
    a baseline already pinned at recall 1.00 leaves no room to show that
    synthetic data helped.

    'sqrt' takes the square root of the inverse-frequency weight (~9x
    instead of ~85x). It is the usual middle ground and keeps the model in
    a regime where it must actually discriminate.

        mode='balanced'  full inverse frequency  (~85x here)
        mode='sqrt'      square root of that     (~9x here)   <- default
        mode='none'      no weighting at all
    """
    counts = np.bincount(y_train, minlength=2).astype(np.float64)
    balanced = counts.sum() / (2.0 * np.maximum(counts, 1.0))

    if mode == "balanced":
        weights = balanced
    elif mode == "sqrt":
        weights = np.sqrt(balanced)
    elif mode == "none":
        weights = np.ones(2, dtype=np.float64)
    else:
        raise ValueError(f"unknown class weight mode: {mode}")

    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def train_one_epoch(model, loader, loss_fn, optimiser):
    """One full pass over the training data, updating the weights."""
    model.train()
    total_loss, n_seen = 0.0, 0

    for windows, labels in loader:
        windows, labels = windows.to(DEVICE), labels.to(DEVICE)

        optimiser.zero_grad()
        logits = model(windows)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(labels)
        n_seen += len(labels)

    return total_loss / max(n_seen, 1)


@torch.no_grad()
def predict(model, loader):
    """
    Run the model over a loader.

    Returns (true labels, predicted labels, seizure probabilities).
    We keep the probabilities because a single hard threshold hides most
    of what the model knows -- see AUPRC in score() below.
    """
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    for windows, labels in loader:
        logits = model(windows.to(DEVICE))
        probs = torch.softmax(logits, dim=1)[:, 1]      # P(seizure)

        all_true.append(labels.numpy())
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        all_prob.append(probs.cpu().numpy())

    return (np.concatenate(all_true),
            np.concatenate(all_pred),
            np.concatenate(all_prob))


def score(y_true, y_prob, threshold, hours):
    """
    Turn predicted probabilities into the numbers we care about.

    THRESHOLD-DEPENDENT (recall, precision, F1, false alarms per hour):
    computed at whatever cut-off you pass in. Easy to read, but they move
    enormously with the cut-off, so always say which one you used.

    THRESHOLD-FREE (AUPRC, average precision): summarises the model across
    EVERY possible cut-off. This is the metric to build your GAN comparison
    on, because it cannot be gamed by nudging the decision threshold. A
    random detector scores roughly the positive rate (~0.016 here).
    """
    y_pred = (y_prob >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    # The number a clinician would ask about: a detector firing 300 times
    # an hour is unusable no matter how good its recall looks.
    fp_per_hour = fp / hours if hours > 0 else float("nan")

    if len(np.unique(y_true)) > 1:
        auprc = average_precision_score(y_true, y_prob)
    else:
        auprc = float("nan")

    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "recall": recall, "precision": precision, "f1": f1,
        "fp_per_hour": fp_per_hour, "auprc": auprc, "threshold": float(threshold),
    }


def pick_threshold(y_calib_true, y_calib_prob):
    """
    Choose the decision threshold using a CALIBRATION recording -- one the
    model never trained on, and which is NOT the recording we report on.

    Why this exists: a fixed 0.5 cut-off is wildly unstable across
    recordings. In testing, the same model and settings produced anywhere
    from 0 to 1748 false alarms per hour depending on which recording was
    held out -- on one fold it labelled every single window a seizure.

    The cause is almost certainly BatchNorm: it stores running mean and
    variance from the training recordings, so a held-out hour with slightly
    different statistics shifts every output score up or down together. The
    RANKING of windows survives this (AUPRC stayed at 0.95 +/- 0.05), but
    the absolute scores drift, so a fixed cut-off lands somewhere different
    each time.

    IMPORTANT -- why not calibrate on the training data itself? We tried
    that first and it failed badly. The model overfits its training
    recordings, so those probabilities saturate near 1.0 and the best
    training threshold comes out around 0.99. Applied to an unseen hour
    whose scores sit lower, it caught nothing at all (recall 0.000). The
    threshold has to be chosen on data the model has NOT memorised, which
    is exactly what the calibration recording is for.

    So each fold uses three disjoint groups of recordings:
        train      -- fit the weights
        calibrate  -- choose the threshold  (unseen during training)
        test       -- report the score      (unseen by both of the above)
    """
    best_threshold, best_f1 = 0.5, -1.0

    # Sweep candidate cut-offs. Using percentiles of the predicted
    # probabilities adapts automatically to whatever scale the model
    # happens to output on this fold.
    candidates = np.unique(np.percentile(y_calib_prob, np.linspace(50, 99.9, 200)))

    for t in candidates:
        pred = (y_calib_prob >= t).astype(np.int64)
        tp = int(((pred == 1) & (y_calib_true == 1)).sum())
        fp = int(((pred == 1) & (y_calib_true == 0)).sum())
        fn = int(((pred == 0) & (y_calib_true == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        if f1 > best_f1:
            best_f1, best_threshold = f1, float(t)

    return best_threshold


def run_fold(X, y, train_idx, calib_idx, test_idx, epochs, n_channels,
             weight_mode="sqrt", norm="group", quiet=False):
    """Train from scratch on train_idx, then predict on test_idx."""
    # Fresh seed each fold so folds are reproducible but not identical.
    torch.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        EEGWindowDataset(X, y, train_idx), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        EEGWindowDataset(X, y, test_idx), batch_size=BATCH_SIZE, shuffle=False
    )

    model = BaselineEEGCNN(n_channels=n_channels, norm=norm).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=make_class_weights(y[train_idx], weight_mode))
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, loss_fn, optimiser)
        if not quiet:
            print(f"      epoch {epoch:2d}/{epochs}  train loss {loss:.4f}")

    # Predictions on the CALIBRATION recording -- not trained on, and not
    # the recording we report. Used only to choose the decision threshold.
    if calib_idx is not None and len(calib_idx) > 0:
        calib_loader = DataLoader(
            EEGWindowDataset(X, y, calib_idx), batch_size=BATCH_SIZE, shuffle=False
        )
        ca_true, _, ca_prob = predict(model, calib_loader)
    else:
        ca_true, ca_prob = None, None

    # NOTE: we evaluate the held-out recording ONCE, after the last epoch.
    # Nothing about it influenced training or the threshold, so this is honest.
    y_true, _, y_prob = predict(model, test_loader)

    return model, y_true, y_prob, ca_true, ca_prob


# ---------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="epochs per fold")
    parser.add_argument("--folds", type=int, default=None,
                        help="only run the first N folds (for quick tests)")
    parser.add_argument("--class-weight", choices=["balanced", "sqrt", "none"],
                        default="sqrt",
                        help="how hard to up-weight the rare seizure class")
    parser.add_argument("--norm", choices=["group", "batch", "none"], default="group",
                        help="normalisation layer; group transfers across recordings")
    parser.add_argument("--skip-final", action="store_true",
                        help="skip training the final all-data model")
    args = parser.parse_args()

    print("=" * 68)
    print("05_train_detector.py -- baseline 1D CNN, leave-one-recording-out CV")
    print("=" * 68)
    print(f"Device: {DEVICE}   epochs per fold: {args.epochs}   "
          f"class weighting: {args.class_weight}   norm: {args.norm}")

    if not DATA_NPZ.exists():
        print(f"ERROR: {DATA_NPZ} not found. Run scripts/04_make_windows.py first.")
        sys.exit(1)

    data = np.load(DATA_NPZ, allow_pickle=False)
    X = data["X"]                       # float16 [n_windows, channels, time]
    y = data["y"]
    metadata = data["metadata"]
    channels = [str(c) for c in data["channels"]]
    sfreq = float(data["sfreq"])
    window_sec = float(data["window_sec"])
    stride_sec = float(data["stride_sec"])

    recordings = metadata["file"]
    n_channels = X.shape[1]

    print(f"\nLoaded {len(y)} windows, shape {X.shape[1:]}, dtype {X.dtype}")
    print(f"  seizure windows : {int(y.sum())}  ({100.0 * y.mean():.2f}%)")
    print(f"  recordings      : {len(np.unique(recordings))}")

    # --- build the folds ------------------------------------------------
    # One fold per recording THAT CONTAINS A SEIZURE. Holding out a
    # recording with no seizures would give a fold where recall is
    # undefined (you cannot measure catching seizures if there are none).
    # Seizure-free recordings still contribute to training in every fold.
    all_recordings = sorted(np.unique(recordings).tolist())
    seizure_recordings = sorted(
        {r for r in all_recordings if y[recordings == r].sum() > 0}
    )

    if len(seizure_recordings) < 2:
        print("\nERROR: need at least 2 recordings with seizures for cross-validation.")
        print(f"Found: {seizure_recordings}")
        print("Download more chb01 files (see scripts/02_download_chbmit.py).")
        sys.exit(1)

    clean_recordings = [r for r in all_recordings if r not in seizure_recordings]
    print(f"  with seizures   : {len(seizure_recordings)} -> {len(seizure_recordings)} folds")
    if clean_recordings:
        print(f"  seizure-free    : {clean_recordings} (always in training)")

    folds = seizure_recordings
    if args.folds is not None:
        folds = folds[: args.folds]
        print(f"  (running only the first {len(folds)} folds)")

    # --- run cross-validation ---------------------------------------------
    print("\n" + "-" * 68)
    print("CROSS-VALIDATION")
    print("-" * 68)

    rows = []
    pooled_true, pooled_prob = [], []

    for k, held_out in enumerate(folds, start=1):
        # Rotate through the seizure recordings to pick a calibration
        # recording that is neither the test recording nor always the same
        # one. It must contain seizures, or the threshold search has no
        # positives to work with.
        others = [r for r in seizure_recordings if r != held_out]
        calib_rec = others[k % len(others)]

        test_idx = np.flatnonzero(recordings == held_out)
        calib_idx = np.flatnonzero(recordings == calib_rec)
        train_idx = np.flatnonzero(
            (recordings != held_out) & (recordings != calib_rec)
        )

        n_pos_test = int(y[test_idx].sum())
        # Each window advances the clock by stride_sec, so this converts
        # a window count into hours of recording time.
        hours = len(test_idx) * stride_sec / 3600.0

        print(f"\n  Fold {k}/{len(folds)}  test on {held_out}, calibrate on {calib_rec}")
        print(f"    train {len(train_idx)} windows ({int(y[train_idx].sum())} seizure) | "
              f"test {len(test_idx)} windows ({n_pos_test} seizure, {hours:.2f} h)")

        _, y_true, y_prob, ca_true, ca_prob = run_fold(
            X, y, train_idx, calib_idx, test_idx, args.epochs, n_channels,
            weight_mode=args.class_weight, norm=args.norm, quiet=True
        )

        # Threshold from the calibration recording -- see pick_threshold().
        thr = pick_threshold(ca_true, ca_prob)

        m = score(y_true, y_prob, thr, hours)
        m_naive = score(y_true, y_prob, 0.5, hours)
        m["recall_at_half"] = m_naive["recall"]
        m["f1_at_half"] = m_naive["f1"]
        m["fp_per_hour_at_half"] = m_naive["fp_per_hour"]
        m["fold"] = k
        m["held_out"] = held_out
        m["calibrated_on"] = calib_rec
        m["n_test_windows"] = len(test_idx)
        m["n_test_seizure"] = n_pos_test
        rows.append(m)

        pooled_true.append(y_true)
        pooled_prob.append(y_prob)

        print(f"    -> AUPRC {m['auprc']:.3f} | at calibrated threshold "
              f"{thr:.3f}: recall {m['recall']:.3f}, precision {m['precision']:.3f}, "
              f"F1 {m['f1']:.3f}, {m['fp']} false alarms ({m['fp_per_hour']:.1f}/h)")
        print(f"       (for comparison, a naive 0.5 cut-off would give "
              f"recall {m_naive['recall']:.3f}, {m_naive['fp_per_hour']:.1f} FA/h)")

    df = pd.DataFrame(rows)[
        ["fold", "held_out", "n_test_windows", "n_test_seizure",
         "calibrated_on", "tp", "fp", "fn", "tn", "auprc", "threshold", "recall", "precision",
         "f1", "fp_per_hour", "recall_at_half", "f1_at_half", "fp_per_hour_at_half"]
    ]

    # --- aggregate ---------------------------------------------------------
    print("\n" + "=" * 68)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 68)
    print()
    show = df[["fold", "held_out", "n_test_seizure", "auprc", "threshold",
               "recall", "precision", "f1", "fp_per_hour",
               "recall_at_half", "fp_per_hour_at_half"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nAcross folds (mean +/- std):")
    for metric in ["auprc", "recall", "precision", "f1", "fp_per_hour"]:
        vals = df[metric].to_numpy(dtype=float)
        print(f"  {metric:<12} {np.nanmean(vals):.3f} +/- {np.nanstd(vals):.3f}"
              f"   (min {np.nanmin(vals):.3f}, max {np.nanmax(vals):.3f})")

    auprc_mean = float(np.nanmean(df["auprc"]))
    auprc_std = float(np.nanstd(df["auprc"]))
    positive_rate = float(y.mean())

    print("\n" + "=" * 68)
    print("YOUR BASELINE")
    print("=" * 68)
    print(f"\n  AUPRC  {auprc_mean:.3f} +/- {auprc_std:.3f}"
          f"   (mean +/- std over {len(folds)} held-out recordings)")
    print(f"         a random detector would score ~{positive_rate:.3f}")
    print(f"\n  At the threshold set on a held-out calibration recording:")
    print(f"    seizure recall  {np.nanmean(df['recall']):.3f} +/- {np.nanstd(df['recall']):.3f}")
    print(f"    false alarms/h  {np.nanmean(df['fp_per_hour']):.1f} +/- {np.nanstd(df['fp_per_hour']):.1f}")

    print("\n  Compare against AUPRC when you test whether synthetic EEG helps.")
    print("  An improvement smaller than the std above is not evidence of anything.")

    # --- why we do NOT pool ---------------------------------------------------
    # Each fold trains a SEPARATE model, and each model has its own
    # probability scale. Concatenating their scores and ranking the result
    # mixes those scales, so the pooled AUPRC comes out LOWER than every
    # individual fold -- an artefact of pooling, not a property of the model.
    # We print it only to show the gap, and to warn you off using it.
    yt = np.concatenate(pooled_true)
    ypr = np.concatenate(pooled_prob)
    if len(np.unique(yt)) > 1:
        pooled_auprc = average_precision_score(yt, ypr)
        print(f"\n  (Pooling all folds' scores together gives AUPRC {pooled_auprc:.3f},")
        print("   which is LOWER than any single fold. That is an artefact: each fold")
        print("   is a different model with its own score scale, so the combined")
        print("   ranking is incoherent. Use the mean-over-folds number above.)")

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved per-fold metrics to: {RESULTS_CSV}")

    # --- final model on all data --------------------------------------------
    # Cross-validation measures how well the METHOD generalises; it leaves
    # you with one model per fold and no single one to keep. So we train once
    # more on everything. It has no honest held-out score of its own -- the
    # CV numbers above are the estimate of how it will behave.
    if not args.skip_final:
        print("\n" + "-" * 68)
        print("Training final model on all recordings ...")
        # Hold one seizure recording out of training purely so we have an
        # unseen recording to set the threshold on.
        calib_rec = seizure_recordings[-1]
        calib_idx = np.flatnonzero(recordings == calib_rec)
        train_idx = np.flatnonzero(recordings != calib_rec)
        print(f"  (training on all but {calib_rec}, which sets the threshold)")

        model, _, _, ca_true, ca_prob = run_fold(
            X, y, train_idx, calib_idx, calib_idx[:BATCH_SIZE],
            args.epochs, n_channels,
            weight_mode=args.class_weight, norm=args.norm, quiet=True
        )
        final_threshold = pick_threshold(ca_true, ca_prob)

        MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "n_channels": n_channels,
                "n_classes": 2,
                "channels": channels,
                "sfreq": sfreq,
                "window_sec": window_sec,
                "stride_sec": stride_sec,
                "time_points": int(X.shape[2]),
                "decision_threshold": final_threshold,
                "class_weight_mode": args.class_weight,
                "norm": args.norm,
                "epochs": args.epochs,
                "cv_auprc_mean": auprc_mean,
                "cv_auprc_std": auprc_std,
                "cv_recall_mean": float(np.nanmean(df["recall"])),
                "cv_fp_per_hour_mean": float(np.nanmean(df["fp_per_hour"])),
                "trained_on": [r for r in all_recordings if r != seizure_recordings[-1]],
                "calibrated_on": seizure_recordings[-1],
            },
            MODEL_OUT,
        )
        print(f"Saved model to: {MODEL_OUT}")
        print(f"  (use decision_threshold={final_threshold:.3f}, not 0.5)")

    print("\nDone.")


if __name__ == "__main__":
    main()