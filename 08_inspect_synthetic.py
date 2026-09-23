"""
08_inspect_synthetic.py
-----------------------
Ask the honest question: did the generator actually learn anything?

A GAN will always produce SOMETHING. The job of this script is to check
whether that something resembles real seizure EEG, using three tests that
get progressively harder to fake.

    TEST 1 -- DOES IT LOOK RIGHT?
        Real and synthetic windows side by side. The weakest test, but if
        it fails here nothing else matters.

    TEST 2 -- DOES IT HAVE THE RIGHT FREQUENCY CONTENT?
        Power spectral density: how much energy sits at each frequency.
        This is the test that catches plausible-looking noise. Real EEG has
        a characteristic 1/f-shaped spectrum, and seizures add power in the
        3-25 Hz range. A generator producing pretty but wrong-frequency
        squiggles is exposed immediately here.

    TEST 3 -- IS IT JUST MEMORISING?
        For each synthetic window, find the most similar REAL window it was
        trained on. Judged against how well real windows match EACH OTHER,
        because that is the only meaningful reference: if the synthetic
        windows match real ones far BETTER than real ones match each other,
        the GAN is memorising; far WORSE, and it is missing real structure.

    TEST 4 -- IS IT DIVERSE ENOUGH?
        How similar are the synthetic windows to each other? Compared
        against how similar the REAL seizure windows are to each other,
        which is the only meaningful reference point.

Inputs:
    models/eeg_cgan.pt                 (from 07_train_gan.py)
    data/processed/chb01_windows.npz   (from 04_make_windows.py)

Output:
    results/synthetic_vs_real.png

Run from the project root:
    python scripts/08_inspect_synthetic.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse the exact model definitions used at training time, so there is no
# chance of the architecture silently drifting between the two scripts.
import importlib
gan_module = importlib.import_module("07_train_gan")
Generator = gan_module.Generator
pick_device = gan_module.pick_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_IN = PROJECT_ROOT / "models" / "eeg_cgan.pt"
DATA_NPZ = PROJECT_ROOT / "data" / "processed" / "chb01_windows.npz"
OUTPUT_PNG = PROJECT_ROOT / "results" / "synthetic_vs_real.png"

N_SYNTHETIC = 256         # how many windows to generate for the statistics
N_SHOW = 2                # how many example windows to draw per class

COLOR_REAL = "#2b6cb0"
COLOR_SYNTH = "#c0392b"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def zscore_clip(X):
    """Same normalisation the GAN was trained on."""
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True)
    return np.clip((X - mean) / (std + 1e-6), -5.0, 5.0)


def mean_psd(windows, sfreq):
    """
    Average power spectral density across windows and channels.

    Welch's method splits each signal into overlapping segments, takes the
    spectrum of each, and averages -- which gives a far less noisy estimate
    than a single FFT of the whole window.
    """
    freqs, psd = signal.welch(
        windows, fs=sfreq, nperseg=256, axis=2
    )                                    # psd: [n_windows, n_channels, n_freqs]
    return freqs, psd.mean(axis=(0, 1)), psd.std(axis=(0, 1))


def plot_window(ax, w, colour, title, spacing=6.0):
    n_ch = w.shape[0]
    for c in range(n_ch):
        ax.plot(w[c] + (n_ch - 1 - c) * spacing, color=colour, linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)


def inter_sample_correlation(windows, n=64):
    """
    How similar are these windows to EACH OTHER, on average?

    Low means variety, high means they all look alike. Crucially, we run
    this on the REAL data too -- the right target is not some fixed number
    but whatever diversity the real seizure windows actually have. Judging
    a generator against an invented threshold tells you nothing.
    """
    n = min(n, len(windows))
    F = windows[:n].reshape(n, -1).astype(np.float32)
    F = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-8)
    C = (F @ F.T) / F.shape[1]
    return float(C[~np.eye(n, dtype=bool)].mean())


def real_to_real_nn_correlation(real, max_n=200):
    """
    THE REFERENCE for the memorisation test.

    For each real window, how well does the MOST SIMILAR OTHER REAL window
    match it? This is the honest yardstick. If two genuine seizure windows
    typically match each other at 0.25, then synthetic windows matching
    real ones at 0.20 are behaving normally -- not "unrelated to EEG",
    which is what a fixed threshold would wrongly conclude.

    (We exclude each window's match with itself, which is trivially 1.0.)
    """
    n = min(max_n, len(real))
    R = real[:n].reshape(n, -1).astype(np.float32)
    R = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-8)
    C = (R @ R.T) / R.shape[1]
    np.fill_diagonal(C, -np.inf)          # ignore self-matches
    return C.max(axis=1)


def nearest_neighbour_correlation(synthetic, real, max_real=1500):
    """
    For each synthetic window, the highest correlation with any real window.

    We flatten each window to a single vector and use Pearson correlation.
    Interpretation depends entirely on the real-to-real reference above.
    """
    rng = np.random.default_rng(0)
    if len(real) > max_real:
        real = real[rng.choice(len(real), max_real, replace=False)]

    S = synthetic.reshape(len(synthetic), -1)
    R = real.reshape(len(real), -1)

    # Standardise each row so a dot product IS the correlation.
    S = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-8)
    R = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-8)

    best = np.empty(len(S))
    chunk = 64
    for i in range(0, len(S), chunk):
        sims = (S[i:i + chunk] @ R.T) / S.shape[1]
        best[i:i + chunk] = sims.max(axis=1)
    return best


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    print("=" * 68)
    print("08_inspect_synthetic.py -- is the synthetic EEG any good?")
    print("=" * 68)

    for path, hint in [(MODEL_IN, "07_train_gan.py"), (DATA_NPZ, "04_make_windows.py")]:
        if not path.exists():
            print(f"ERROR: {path} not found. Run scripts/{hint} first.")
            sys.exit(1)

    device = pick_device(args.device)
    ckpt = torch.load(MODEL_IN, map_location=device, weights_only=False)

    G = Generator(n_channels=ckpt["n_channels"],
                  latent_dim=ckpt["latent_dim"],
                  base=ckpt["base_channels"]).to(device)
    G.load_state_dict(ckpt["generator"])
    G.eval()

    sfreq = ckpt["sfreq"]
    print(f"Loaded generator trained for {ckpt['epochs']} epochs on "
          f"[{ckpt['n_channels']}, {ckpt['time_points']}] windows.")

    # --- real data ------------------------------------------------------
    data = np.load(DATA_NPZ, allow_pickle=False)
    X, y = data["X"], data["y"]

    real_seizure = zscore_clip(X[y == 1].astype(np.float32))
    real_background = zscore_clip(
        X[y == 0][:N_SYNTHETIC].astype(np.float32)
    )
    print(f"Real seizure windows available: {len(real_seizure)}")

    # --- generate --------------------------------------------------------
    with torch.no_grad():
        z = torch.randn(N_SYNTHETIC, ckpt["latent_dim"], device=device)
        lab_seizure = torch.ones(N_SYNTHETIC, dtype=torch.long, device=device)
        synth_seizure = G(z, lab_seizure).cpu().numpy()

        z2 = torch.randn(N_SYNTHETIC, ckpt["latent_dim"], device=device)
        lab_bg = torch.zeros(N_SYNTHETIC, dtype=torch.long, device=device)
        synth_background = G(z2, lab_bg).cpu().numpy()

    print(f"Generated {N_SYNTHETIC} synthetic seizure windows.")

    # --- TEST 2: spectra ---------------------------------------------------
    f_real, psd_real, sd_real = mean_psd(real_seizure, sfreq)
    f_syn, psd_syn, sd_syn = mean_psd(synth_seizure, sfreq)
    _, psd_bg_real, _ = mean_psd(real_background, sfreq)
    _, psd_bg_syn, _ = mean_psd(synth_background, sfreq)

    band = (f_real >= 1) & (f_real <= 40)
    # Compare the two spectra on a log scale, where EEG power naturally lives.
    log_diff = np.abs(np.log10(psd_syn[band] + 1e-12)
                      - np.log10(psd_real[band] + 1e-12))
    spectral_error = float(log_diff.mean())

    # --- TEST 3: memorisation ----------------------------------------------
    nn_corr = nearest_neighbour_correlation(synth_seizure, real_seizure)
    real_nn = real_to_real_nn_correlation(real_seizure)

    # --- report ------------------------------------------------------------
    print("\n" + "-" * 68)
    print("RESULTS")
    print("-" * 68)

    print(f"\nTEST 2 -- spectral match (1-40 Hz)")
    print(f"  mean |log10 power difference| : {spectral_error:.3f}")
    print("    below ~0.3  the spectra are a close match")
    print("    0.3 to 0.7  roughly the right shape, wrong magnitudes")
    print("    above ~0.7  the generator has not learned EEG frequency structure")

    print(f"\nTEST 3 -- memorisation check")
    print(f"  REAL-to-real nearest neighbour      : mean {real_nn.mean():.3f}, "
          f"max {real_nn.max():.3f}   <- the reference")
    print(f"  SYNTHETIC-to-real nearest neighbour : mean {nn_corr.mean():.3f}, "
          f"max {nn_corr.max():.3f}")
    if nn_corr.max() > 0.9:
        print("    VERDICT: copying training data. Augmentation would just")
        print("    duplicate windows you already have.")
    elif nn_corr.mean() > real_nn.mean() + 0.15:
        print("    VERDICT: closer to real windows than real windows are to")
        print("    each other -- drifting towards memorisation.")
    elif nn_corr.mean() < real_nn.mean() - 0.15:
        print("    VERDICT: NOT close enough. Synthetic windows resemble real")
        print("    seizures less than real seizures resemble each other, so")
        print("    they are probably missing real structure.")
    else:
        print("    VERDICT: synthetic windows sit at about the same distance")
        print("    from real data as real windows sit from each other. Good --")
        print("    new samples, same family.")

    # --- TEST 4: diversity, measured against the real data --------------
    synth_div = inter_sample_correlation(synth_seizure)
    real_div = inter_sample_correlation(real_seizure)

    print(f"\nTEST 4 -- mode collapse / diversity")
    print(f"  correlation between REAL seizure windows      : {real_div:.3f}  <- the target")
    print(f"  correlation between SYNTHETIC seizure windows : {synth_div:.3f}")
    if synth_div > real_div + 0.25:
        print(f"    VERDICT: too uniform. The generator is producing variations on")
        print(f"    a narrow set of windows rather than covering the real range.")
    elif synth_div > real_div + 0.10:
        print(f"    VERDICT: somewhat too uniform, but in the right region.")
    else:
        print(f"    VERDICT: diversity comparable to real data. Good.")
    print("    (A fixed threshold would be meaningless here -- what counts as")
    print("     'diverse' depends entirely on how varied the real seizures are.)")

    # --- figure --------------------------------------------------------------
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.3, wspace=0.25)

    for k in range(N_SHOW):
        plot_window(fig.add_subplot(gs[0, k]), real_seizure[
            len(real_seizure) // 2 + k], COLOR_REAL,
            f"REAL seizure #{k + 1}")
    for k in range(N_SHOW):
        plot_window(fig.add_subplot(gs[0, N_SHOW + k]), synth_seizure[k],
                    COLOR_SYNTH, f"SYNTHETIC seizure #{k + 1}")

    # PSD comparison
    ax_psd = fig.add_subplot(gs[1, :2])
    ax_psd.semilogy(f_real[band], psd_real[band], color=COLOR_REAL,
                    linewidth=1.8, label="real seizure")
    ax_psd.fill_between(f_real[band],
                        np.maximum(psd_real[band] - sd_real[band], 1e-6),
                        psd_real[band] + sd_real[band],
                        color=COLOR_REAL, alpha=0.15)
    ax_psd.semilogy(f_syn[band], psd_syn[band], color=COLOR_SYNTH,
                    linewidth=1.8, label="synthetic seizure")
    ax_psd.fill_between(f_syn[band],
                        np.maximum(psd_syn[band] - sd_syn[band], 1e-6),
                        psd_syn[band] + sd_syn[band],
                        color=COLOR_SYNTH, alpha=0.15)
    ax_psd.semilogy(f_real[band], psd_bg_real[band], color=COLOR_REAL,
                    linewidth=1.0, linestyle="--", alpha=0.7,
                    label="real background")
    ax_psd.set_xlabel("frequency (Hz)")
    ax_psd.set_ylabel("power spectral density")
    ax_psd.set_title(f"Frequency content -- the real test\n"
                     f"mean |log10 difference| = {spectral_error:.3f}",
                     fontsize=10)
    ax_psd.legend(fontsize=8)
    ax_psd.grid(alpha=0.25, linewidth=0.5)
    for s in ("top", "right"):
        ax_psd.spines[s].set_visible(False)

    # Memorisation histogram
    ax_nn = fig.add_subplot(gs[1, 2])
    ax_nn.hist(nn_corr, bins=30, color=COLOR_SYNTH, alpha=0.75, label="synthetic")
    ax_nn.hist(real_nn, bins=30, color=COLOR_REAL, alpha=0.55, label="real (reference)")
    ax_nn.axvline(0.9, color="black", linestyle="--", linewidth=1.2)
    ax_nn.legend(fontsize=8)
    ax_nn.set_xlabel("correlation with closest real window")
    ax_nn.set_ylabel("count")
    ax_nn.set_title("Distance to nearest real window\n(overlap with blue = right family)",
                    fontsize=9)
    for s in ("top", "right"):
        ax_nn.spines[s].set_visible(False)

    # Amplitude distribution
    ax_amp = fig.add_subplot(gs[1, 3])
    ax_amp.hist(real_seizure.ravel()[::37], bins=60, density=True,
                color=COLOR_REAL, alpha=0.55, label="real")
    ax_amp.hist(synth_seizure.ravel()[::37], bins=60, density=True,
                color=COLOR_SYNTH, alpha=0.55, label="synthetic")
    ax_amp.set_xlabel("amplitude (standard deviations)")
    ax_amp.set_ylabel("density")
    ax_amp.set_title("Amplitude distribution", fontsize=10)
    ax_amp.legend(fontsize=8)
    for s in ("top", "right"):
        ax_amp.spines[s].set_visible(False)

    fig.suptitle("NeuroSynth -- are the synthetic seizure windows realistic?",
                 fontsize=13)

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figure to: {OUTPUT_PNG}")
    print("Done.")


if __name__ == "__main__":
    main()