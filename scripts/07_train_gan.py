"""
07_train_gan.py
---------------
Train a CONDITIONAL GAN that generates synthetic 4-second EEG windows.

--------------------------------------------------------------------
WHAT A GAN IS, IN ONE PARAGRAPH
--------------------------------------------------------------------
Two networks compete. The GENERATOR takes random noise and tries to
produce something that looks like real EEG. The CRITIC (also called the
discriminator) looks at a window and scores how real it seems. They train
against each other: the critic gets better at spotting fakes, which forces
the generator to make better fakes, and so on. Neither ever "wins" -- you
stop when the samples look good enough.

CONDITIONAL means the generator also receives a LABEL saying which kind of
window to produce (0 = non-seizure, 1 = seizure). After training you can
ask it specifically for seizure windows, which is the whole point here:
you have only 230 real seizure windows and 13,524 non-seizure ones, and
that imbalance is what limits the detector.

--------------------------------------------------------------------
DESIGN CHOICES, AND WHY
--------------------------------------------------------------------
1. TRAINED ON BOTH CLASSES, NOT JUST SEIZURES.
   Training on 230 seizure windows alone almost guarantees mode collapse
   (the generator memorises a few windows and outputs slight variants).
   By conditioning, the generator learns what EEG looks like in general
   from the abundant non-seizure data, and uses the label to specialise.

2. HINGE LOSS + SPECTRAL NORMALISATION, not WGAN-GP.
   Both stabilise GAN training. Spectral norm does it by constraining the
   critic's weights directly, which costs almost nothing. WGAN-GP does it
   with a gradient penalty that needs a second backward pass through the
   critic every step -- roughly 2-3x slower. On a laptop CPU that matters.

3. FiLM CONDITIONING THROUGHOUT THE GENERATOR.
   The noise modulates every layer, not just the first. Without this the
   GroupNorm layers scrubbed the noise away and the generator produced the
   same window every time. See the FiLMUpBlock class.

4. MINIBATCH STANDARD DEVIATION IN THE CRITIC.
   Added after a first attempt collapsed: 40 epochs of training produced
   samples with 0.993 correlation to each other, i.e. one window repeated.
   See the MinibatchStdDev class for why this fixes it.

5. GROUPNORM IN THE GENERATOR, NONE IN THE CRITIC.
   Same reasoning as the detector: BatchNorm's running statistics make
   outputs depend on what else is in the batch. In a critic it also
   weakens the spectral-norm guarantee.

6. WE WORK IN Z-SCORED SPACE.
   Windows are standardised per channel before training, exactly as the
   detector does. So the generator's output is already in the detector's
   input space, and augmentation later needs no conversion.

Input:
    data/processed/chb01_windows.npz

Outputs:
    models/eeg_cgan.pt                 generator + critic weights
    results/gan_loss_curves.png        training curves
    results/gan_samples_epoch###.png   samples as training progresses

Run from the project root:
    python scripts/07_train_gan.py                 # ~1 hour target
    python scripts/07_train_gan.py --quick         # ~5 min sanity check
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_NPZ = PROJECT_ROOT / "data" / "processed" / "chb01_windows.npz"
MODEL_OUT = PROJECT_ROOT / "models" / "eeg_cgan.pt"
RESULTS_DIR = PROJECT_ROOT / "results"

LATENT_DIM = 100          # size of the random noise vector fed to the generator
LABEL_EMBED_DIM = 16      # size of the learned vector representing each class
BASE_CHANNELS = 64        # width of the networks; raise for capacity, lower for speed

BATCH_SIZE = 64
EPOCHS = 60
LR_G = 2e-4
LR_D = 2e-4
BETAS = (0.5, 0.9)        # Adam betas; low beta1 is standard practice for GANs
N_CRITIC = 2              # critic updates per generator update

# We do not need all 13,524 non-seizure windows to teach the generator what
# background EEG looks like, and using all of them makes every epoch slow.
# We keep all seizure windows and a random sample of non-seizure ones.
MAX_NON_SEIZURE = 2000

SAMPLE_EVERY = 10         # save a figure of samples every N epochs
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def pick_device(requested: str = "auto") -> torch.device:
    """
    Choose where to run.

    Apple Silicon Macs have a GPU that PyTorch can use through the 'mps'
    backend, and it is typically several times faster than the CPU for
    this kind of model. We prefer CUDA, then MPS, then CPU.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------

class ZScoredWindows(Dataset):
    """
    Serves z-scored EEG windows plus their labels.

    Identical normalisation to the detector: subtract each channel's mean
    and divide by its standard deviation, within each window. This puts
    every window on a comparable scale, which both the GAN and the
    detector need.
    """

    def __init__(self, X, y, indices):
        self.X, self.y, self.indices = X, y, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        w = torch.from_numpy(self.X[idx].astype(np.float32))

        mean = w.mean(dim=1, keepdim=True)
        std = w.std(dim=1, keepdim=True)
        w = (w - mean) / (std + 1e-6)

        # Clip extreme values. Real EEG contains artefacts up to ~1800 uV;
        # after z-scoring those become huge spikes that a GAN will happily
        # learn to reproduce instead of learning brain activity.
        w = torch.clamp(w, -5.0, 5.0)

        return w, int(self.y[idx])


# ---------------------------------------------------------------------
# 3. Generator
# ---------------------------------------------------------------------

class FiLMUpBlock(nn.Module):
    """
    One upsampling block whose behaviour is steered by the noise vector.

    WHY THIS IS NOT JUST A CONV BLOCK -- this is the fix for our second
    collapse. In the first design the noise entered only at the very start,
    through a linear layer, and every block afterwards applied GroupNorm.
    GroupNorm removes each sample's scale and offset, which is precisely
    how the noise was expressing itself, so its influence was scrubbed away
    layer by layer. Measured on the trained model, changing z moved the
    output by only 9% of the signal's own variation -- the generator had
    effectively become a constant function.

    FiLM (Feature-wise Linear Modulation) fixes this by re-injecting the
    noise AFTER each normalisation, as a per-feature scale and shift:

        normalise  ->  multiply by (1 + gamma)  ->  add beta  ->  activate

    where gamma and beta are computed from the noise. Now the noise cannot
    be normalised away, because it is applied after the normalisation at
    every depth. Note the GroupNorm below uses affine=False: FiLM supplies
    the scale and shift, so a second learned pair would be redundant.
    """

    def __init__(self, in_ch, out_ch, w_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        # Upsample-then-convolve rather than ConvTranspose1d: transposed
        # convolutions produce periodic "checkerboard" artefacts, which in a
        # time series would be a fake rhythm the detector could latch onto.
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=9, padding=4)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch),
                                 num_channels=out_ch, affine=False)
        self.film = nn.Linear(w_dim, out_ch * 2)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        x = self.conv(self.up(x))
        x = self.norm(x)
        gamma, beta = self.film(w).chunk(2, dim=1)
        x = x * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        return self.act(x)


class Generator(nn.Module):
    """
    noise + label  ->  a [18, 1024] EEG window.

    Shape walkthrough for a batch of 64:
        noise                     [64, 100]
        label embedding           [64, 16]
        combined into a "style"   [64, 116]  -- steers every block via FiLM
        linear + reshape          [64, 256, 64]     start short and wide
        FiLM up block 1           [64, 128, 128]
        FiLM up block 2           [64,  64, 256]
        FiLM up block 3           [64,  32, 512]
        FiLM up block 4           [64,  16, 1024]
        final conv                [64,  18, 1024]   one row per EEG channel

    Each block doubles the time axis, so 64 -> 1024 samples, which at
    256 Hz is exactly our 4-second window.
    """

    def __init__(self, n_channels=18, latent_dim=LATENT_DIM, base=BASE_CHANNELS):
        super().__init__()
        self.latent_dim = latent_dim

        # A learned vector for each class, so the generator can steer its
        # output towards seizure-like or background-like activity.
        self.label_embed = nn.Embedding(2, LABEL_EMBED_DIM)

        self.w_dim = latent_dim + LABEL_EMBED_DIM
        self.start_len = 64
        self.start_ch = base * 4

        self.fc = nn.Linear(self.w_dim, self.start_ch * self.start_len)

        self.block1 = FiLMUpBlock(self.start_ch, base * 2, self.w_dim)
        self.block2 = FiLMUpBlock(base * 2, base, self.w_dim)
        self.block3 = FiLMUpBlock(base, base // 2, self.w_dim)
        self.block4 = FiLMUpBlock(base // 2, base // 4, self.w_dim)

        self.to_signal = nn.Conv1d(base // 4, n_channels, kernel_size=9, padding=4)

    def forward(self, z, labels):
        emb = self.label_embed(labels)                       # [B, 16]
        w = torch.cat([z, emb], dim=1)                       # [B, 116] "style"

        x = self.fc(w).view(-1, self.start_ch, self.start_len)
        x = self.block1(x, w)
        x = self.block2(x, w)
        x = self.block3(x, w)
        x = self.block4(x, w)
        x = self.to_signal(x)

        # tanh bounds the output; scaling by 4 matches the range of real
        # z-scored EEG (roughly +/- 4 standard deviations after clipping).
        return torch.tanh(x) * 4.0


# ---------------------------------------------------------------------
# 4. Critic
# ---------------------------------------------------------------------

class MinibatchStdDev(nn.Module):
    """
    The standard fix for mode collapse -- and the reason it works is neat.

    A critic that only ever sees ONE window at a time cannot tell the
    difference between a generator producing endless variety and one
    producing the same window over and over. Each individual sample looks
    equally plausible, so there is no gradient pushing towards diversity.

    This layer computes how much the samples in the batch VARY from each
    other, and hands that number to the critic as an extra feature. Now a
    batch of 64 identical fakes has a variety score of ~0 while a batch of
    64 real windows has a high one, so the critic spots collapse instantly
    and the generator is forced to spread out.
    """

    def forward(self, x):
        # x: [batch, channels, time]
        std = x.std(dim=0, keepdim=True)                 # variation across the batch
        mean_std = std.mean().expand(x.size(0), 1, x.size(2))
        return torch.cat([x, mean_std], dim=1)


class Critic(nn.Module):
    """
    an EEG window + its label  ->  one number: how real does this look?

    The label enters as an extra input channel, broadcast across time. That
    is the simplest conditioning scheme to explain: the critic sees both the
    signal and which class it claims to be, so the generator cannot cheat by
    producing a perfect non-seizure window when asked for a seizure.

    Every convolution is wrapped in spectral_norm, which caps how sharply
    the critic's output can change. That is what keeps training stable.
    """

    def __init__(self, n_channels=18, base=BASE_CHANNELS):
        super().__init__()

        def down_block(in_ch, out_ch):
            return nn.Sequential(
                spectral_norm(nn.Conv1d(in_ch, out_ch, kernel_size=9,
                                        stride=4, padding=4)),
                nn.LeakyReLU(0.2),
            )

        self.blocks = nn.Sequential(
            down_block(n_channels + 1, base // 2),   # 1024 -> 256
            down_block(base // 2, base),             # 256  -> 64
            down_block(base, base * 2),              # 64   -> 16
            down_block(base * 2, base * 4),          # 16   -> 4
        )
        self.mb_std = MinibatchStdDev()
        # +1 channel for the batch-variety feature added by MinibatchStdDev
        self.out = spectral_norm(nn.Linear((base * 4 + 1) * 4, 1))

    def forward(self, x, labels):
        # Broadcast the label to a full-length channel of 0.0 or 1.0.
        lab = labels.view(-1, 1, 1).float().expand(-1, 1, x.shape[2])
        x = torch.cat([x, lab], dim=1)               # [B, 19, 1024]
        x = self.blocks(x)
        x = self.mb_std(x)
        return self.out(x.flatten(1)).squeeze(1)


# ---------------------------------------------------------------------
# 5. Plotting helpers
# ---------------------------------------------------------------------

def save_sample_figure(generator, device, epoch, path, n_show=6):
    """Draw a few generated seizure windows so you can watch progress."""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_show, LATENT_DIM, device=device)
        labels = torch.ones(n_show, dtype=torch.long, device=device)   # seizure
        fake = generator(z, labels).cpu().numpy()
    generator.train()

    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 6), sharey=True)
    spacing = 6.0
    for k, ax in enumerate(axes):
        w = fake[k]
        for c in range(w.shape[0]):
            ax.plot(w[c] + (w.shape[0] - 1 - c) * spacing,
                    color="#c0392b", linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"sample {k + 1}", fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle(f"Generated SEIZURE windows -- epoch {epoch}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_loss_curves(history, path):
    """
    Plot the two losses over time.

    Reading GAN curves is not like reading a normal training curve. You are
    NOT looking for both to go to zero -- that would mean one network beat
    the other and learning stopped. You want them to stay in a rough
    equilibrium, wobbling around without either running away.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(history["epoch"], history["d_loss"], label="critic loss",
            color="#2b6cb0", linewidth=1.4)
    ax.plot(history["epoch"], history["g_loss"], label="generator loss",
            color="#c0392b", linewidth=1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("GAN training -- healthy training keeps these in balance,\n"
                 "neither collapsing to zero nor diverging", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.25, linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default="auto",
                        help="auto | cpu | mps | cuda")
    parser.add_argument("--quick", action="store_true",
                        help="tiny run to check everything works (~5 min)")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 3

    device = pick_device(args.device)

    print("=" * 68)
    print("07_train_gan.py -- conditional GAN for synthetic EEG windows")
    print("=" * 68)
    print(f"Device: {device}")
    if device.type == "cpu":
        print("  (no GPU found -- this will be slow. On an Apple Silicon Mac,")
        print("   'mps' should be picked up automatically.)")

    # --- load data -----------------------------------------------------
    if not DATA_NPZ.exists():
        print(f"ERROR: {DATA_NPZ} not found. Run scripts/04_make_windows.py first.")
        sys.exit(1)

    data = np.load(DATA_NPZ, allow_pickle=False)
    X, y = data["X"], data["y"]
    n_channels = X.shape[1]
    time_points = X.shape[2]

    seizure_idx = np.flatnonzero(y == 1)
    non_seizure_idx = np.flatnonzero(y == 0)

    # Keep every seizure window, subsample the rest.
    rng = np.random.default_rng(RANDOM_SEED)
    keep_non = rng.choice(non_seizure_idx,
                          size=min(MAX_NON_SEIZURE, len(non_seizure_idx)),
                          replace=False)
    train_idx = np.concatenate([seizure_idx, keep_non])
    rng.shuffle(train_idx)

    print(f"\nTraining windows : {len(train_idx)}")
    print(f"  seizure        : {len(seizure_idx)}  (all of them)")
    print(f"  non-seizure    : {len(keep_non)}  (random sample of {len(non_seizure_idx)})")
    print(f"  window shape   : [{n_channels}, {time_points}]")

    dataset = ZScoredWindows(X, y, train_idx)

    # Balanced sampling: without this, ~90% of every batch would be
    # non-seizure and the generator would barely learn the seizure class.
    labels_in_train = y[train_idx]
    class_count = np.bincount(labels_in_train, minlength=2)
    sample_weights = (1.0 / class_count)[labels_in_train]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_idx),
        replacement=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        drop_last=True)
    print(f"  balanced batches: each batch is ~50/50 seizure vs non-seizure")

    # --- build models ----------------------------------------------------
    G = Generator(n_channels=n_channels).to(device)
    D = Critic(n_channels=n_channels).to(device)

    print(f"\nGenerator parameters : {sum(p.numel() for p in G.parameters()):,}")
    print(f"Critic parameters    : {sum(p.numel() for p in D.parameters()):,}")

    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=BETAS)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=BETAS)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    history = {"epoch": [], "d_loss": [], "g_loss": []}

    print("\n" + "-" * 68)
    print("TRAINING")
    print("-" * 68)
    print("Watch the two losses: they should stay in a rough balance.")
    print("If the critic loss crashes to 0, it has beaten the generator and")
    print("learning has stalled.\n")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        d_running, g_running, n_batches = 0.0, 0.0, 0

        for real, labels in loader:
            real = real.to(device)
            labels = labels.to(device)
            batch = real.size(0)

            # ---------------- train the critic ----------------
            # Hinge loss: push real scores above +1 and fake scores below -1.
            # Anything already past those margins contributes no gradient,
            # which stops the critic from over-training on easy examples.
            for _ in range(N_CRITIC):
                z = torch.randn(batch, LATENT_DIM, device=device)
                with torch.no_grad():
                    fake = G(z, labels)

                d_real = D(real, labels)
                d_fake = D(fake, labels)

                d_loss = (F.relu(1.0 - d_real).mean()
                          + F.relu(1.0 + d_fake).mean())

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            # ---------------- train the generator ----------------
            # The generator simply wants the critic's score on its output
            # to be as high as possible.
            z = torch.randn(batch, LATENT_DIM, device=device)
            fake = G(z, labels)
            g_loss = -D(fake, labels).mean()

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            d_running += d_loss.item()
            g_running += g_loss.item()
            n_batches += 1

        d_avg = d_running / max(n_batches, 1)
        g_avg = g_running / max(n_batches, 1)
        history["epoch"].append(epoch)
        history["d_loss"].append(d_avg)
        history["g_loss"].append(g_avg)

        elapsed = time.time() - start_time
        per_epoch = elapsed / epoch
        remaining = per_epoch * (args.epochs - epoch)
        print(f"epoch {epoch:3d}/{args.epochs} | critic {d_avg:7.4f} | "
              f"generator {g_avg:8.4f} | {per_epoch:5.1f}s/epoch | "
              f"~{remaining / 60:5.1f} min left")

        if epoch % SAMPLE_EVERY == 0 or epoch == args.epochs:
            save_sample_figure(G, device, epoch,
                               RESULTS_DIR / f"gan_samples_epoch{epoch:03d}.png")

    # --- save ------------------------------------------------------------
    save_loss_curves(history, RESULTS_DIR / "gan_loss_curves.png")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator": G.state_dict(),
            "critic": D.state_dict(),
            "latent_dim": LATENT_DIM,
            "n_channels": n_channels,
            "time_points": time_points,
            "base_channels": BASE_CHANNELS,
            "epochs": args.epochs,
            "channels": [str(c) for c in data["channels"]],
            "sfreq": float(data["sfreq"]),
            "history": history,
        },
        MODEL_OUT,
    )

    total_min = (time.time() - start_time) / 60
    print(f"\nTrained for {total_min:.1f} minutes.")
    print(f"Saved model to      : {MODEL_OUT}")
    print(f"Saved loss curves to: {RESULTS_DIR / 'gan_loss_curves.png'}")
    print(f"Saved samples to    : {RESULTS_DIR}/gan_samples_epoch*.png")
    print("\nNext: python scripts/08_inspect_synthetic.py")


if __name__ == "__main__":
    main()
