# NeuroSynth

An EEG seizure detection pipeline built on the CHB-MIT dataset, and a controlled experiment testing whether GAN-generated synthetic seizure data improves detection under severe class imbalance.

The goal is not to build a deployable clinical tool. It is to answer one question carefully, with an evaluation setup that I would be willing to defend, and to report the answer honestly even though the answer turned out to be negative.

## Research Question

Seizures make up a very small fraction of any EEG recording. In the data I used, only 1.67% of windows contain seizure activity. This imbalance is the central obstacle to training a detector: there simply are not many positive examples to learn from.

A commonly proposed fix is to generate synthetic positives with a generative model and add them to the training set. I wanted to test whether that actually works, rather than assume it does.

The comparison is direct: train the same detector twice, once on real data alone and once on real data plus synthetic seizure windows, and evaluate both on the same held-out recording.

## Dataset

CHB-MIT Scalp EEG Database (PhysioNet), collected at Boston Children's Hospital. Every seizure is annotated to the second by neurologists.

I used patient `chb01`:

- 8 one-hour recordings — the 7 that contain seizures, plus one seizure-free hour
- 7 seizures totalling 442 seconds
- 18 bipolar channels, sampled at 256 Hz
- 13,754 windows after preprocessing, of which 230 are seizure

That works out to a **1.67% positive rate**.

I deliberately included one seizure-free recording so the detector's false alarm behaviour could be observed on a completely normal hour.

## Example

Running the detector produces per-fold results like this:

```
Fold 3/7  test on chb01_15.edf, calibrate on chb01_18.edf
  train 10156 windows (163 seizure) | test 1799 windows (21 seizure, 1.00 h)
  -> AUPRC 0.566 | at calibrated threshold 0.092: recall 0.810,
     precision 0.425, F1 0.557, 23 false alarms (23.0/h)
```

Each fold holds out one entire hour of recording, trains on the rest, and reports how well the model found the seizures in that hour.

## Main Pipeline

```
raw EDF recordings
  -> parse neurologist seizure annotations into a table
  -> filter, window, and label the signal
  -> train a 1D CNN detector, evaluate with recording-level cross-validation
  -> train a conditional GAN on the same windows
  -> validate the synthetic output against the real data
  -> retrain the detector with synthetic data added, compare head to head
```

## Project Structure

```
neurosynth/
├── scripts/
│   ├── 01_check_setup.py            environment and dependency check
│   ├── 02_download_chbmit.py        fetch recordings from PhysioNet
│   ├── 03_parse_summary.py          seizure annotations -> CSV
│   ├── 04_make_windows.py           EDF -> filtered, labelled windows
│   ├── 05_train_detector.py         detector + cross-validation
│   ├── 06_visualize_windows.py      sanity-check figure
│   ├── 07_train_gan.py              conditional GAN
│   ├── 08_inspect_synthetic.py      synthetic data validation
│   └── 09_augment_and_evaluate.py   the augmentation experiment
├── results/                         figures and per-fold metric tables
├── data/                            raw and processed data (not in repo)
├── models/                          trained weights (not in repo)
└── requirements.txt
```

`data/` and `models/` are excluded from the repository. The processed dataset is roughly 468 MB and both are fully regenerable by running the scripts in order.

## Methods

### Preprocessing

Each recording is band-pass filtered between 0.5 and 40 Hz. Below 0.5 Hz is mostly electrode drift and sweat artefact; above 40 Hz is mostly muscle activity and mains noise. The clinically relevant content sits in between.

The filtered signal is cut into 4-second windows with a 2-second stride, giving 1024 samples per channel per window, so each window is a `[18, 1024]` array. A window is labelled seizure if it overlaps an annotated seizure interval at all.

Every window is z-scored per channel. EEG amplitude varies with electrode contact quality and skull thickness, so standardising forces the model to learn from waveform shape rather than absolute voltage.

One dataset quirk worth noting: the `chb01` EDF headers list the channel label `T8-P8` twice, and MNE de-duplicates them into `T8-P8-0` and `T8-P8-1`. Selecting channels naively raises an error, so channel names are normalised before matching.

### Detector

A small 1D convolutional network: three convolution blocks, global average pooling, and a linear classifier. Roughly 20,000 parameters.

The small size is intentional. With only 230 positive examples, a larger model memorises them rather than learning anything that generalises.

Class weighting uses the square root of inverse frequency rather than full inverse frequency. At full weighting the positive class receives roughly 85 times the weight, and the model responds by flagging most of the recording — recall of 1.000 with precision of 0.015, and over 1,100 false alarms per hour. Technically high recall, practically useless.

### Evaluation protocol

This took the most iteration and is the part I would most want someone to scrutinise.

**Leave-one-recording-out cross-validation.** Each fold holds out an entire one-hour recording. My first version used a random window-level split, which reported a seizure F1 of 1.000. That number was an artefact: windows overlap by 50%, so a random split placed windows sharing two full seconds of identical signal on both sides of the divide. The model was recognising data it had effectively already seen.

**Fixed epoch count.** The same first version trained for 15 epochs and reported whichever epoch scored best on the held-out data. Selecting the maximum of a noisy sequence and reporting it as the result is guaranteed to overstate. The current version trains for a fixed number of epochs and reports the final one.

**A separate calibration recording.** Each fold partitions recordings three ways: six for training, one for setting the decision threshold, and one for reporting. The calibration split exists because choosing the threshold on training data failed badly — the model overfits its training recordings, so those probabilities saturate near 1.0 and the optimal threshold came out at 0.991. Applied to an unseen hour, it caught nothing.

**GroupNorm instead of BatchNorm.** With BatchNorm, false alarms per hour across the seven folds were 11, 2, 0, 348, 153, 0, and 1748, while AUPRC stayed steady around 0.97. Stable ranking alongside unstable thresholds pointed at the normalisation layer: BatchNorm reuses statistics accumulated during training, so a held-out recording with different characteristics shifts every output score together. Order survives, absolute values do not. GroupNorm normalises each sample from its own statistics, which removed the drift and roughly halved the variance of every metric.

### Conditional GAN

The generator takes a 100-dimensional noise vector plus a class label and produces a `[18, 1024]` window. Four upsampling blocks take the time axis from 64 to 1024 samples. The critic takes a window with the label broadcast as an additional channel and outputs a single realness score.

Training uses hinge loss with spectral normalisation rather than WGAN-GP. Both constrain the critic, but spectral normalisation rescales the weight matrices directly, whereas a gradient penalty requires a second backward pass on every step.

The generator upsamples and then convolves rather than using a transposed convolution. Transposed convolutions produce periodic checkerboard artefacts, which in a time series would appear as a spurious rhythm — exactly the kind of signal a seizure detector might latch onto, which would have invalidated the experiment.

**Trained on both classes, not only the seizure windows.** Most of what makes EEG look like EEG — amplitude range, spectral shape, inter-channel structure — is shared between seizure and background activity, and can be learned from the abundant negative windows. Conditioning lets the label carry only the seizure-specific difference. Training on 230 windows alone would almost certainly have collapsed.

**FiLM conditioning.** An earlier version injected the noise only at the first layer, and every subsequent block applied GroupNorm. Measuring the trained model showed that changing the noise vector moved the output by just 9% of the signal's own variation — the generator had become close to a constant function. Normalisation was removing the per-sample scale and shift through which the noise was expressing itself. FiLM re-injects the conditioning after each normalisation, as a per-feature scale and shift, so it cannot be scrubbed away. This raised the noise's influence to 35% and resolved the collapse.

### Validating the synthetic data

There is no accuracy to compute for a generated window, so I used three checks, each judged against a reference measured from the real data rather than a threshold I chose myself.

- **Spectral fidelity.** Mean absolute log-difference between real and synthetic power spectral density over 1–40 Hz, computed with Welch's method.
- **Memorisation.** For each synthetic window, the highest correlation with any real window — compared against how well real windows match each other.
- **Diversity.** Mean pairwise correlation among synthetic samples — compared against the same statistic computed on real seizure windows.

The reference matters more than it might appear. I initially judged the last two against thresholds picked by intuition, and on real data that produced a false failure: a nearest-neighbour correlation of 0.204 looked alarming until I computed the real-to-real figure and found it was 0.212. The synthetic windows were behaving exactly like real ones. The model never changed, only the yardstick did.

## Evaluation

**AUPRC (average precision) is the primary metric.** At a 1.67% positive rate, accuracy is meaningless — a model that always predicts "no seizure" scores 98.3%. Precision, recall and F1 are all sensitive to where the decision threshold is placed, which makes them fragile for comparing experiments. AUPRC summarises the model across every possible threshold, so it measures whether seizure windows are ranked above background regardless of where the cut is made. A random detector would score roughly 0.017 on this data.

Secondary metrics are recall, precision, and false alarms per hour at the calibrated threshold. False alarms per hour is included because it is the number a clinician would actually care about; a detector firing hundreds of times an hour is unusable whatever its recall.

## Results

### Baseline detector

| metric | value |
|---|---|
| AUPRC | 0.898 ± 0.138 |
| seizure recall | 0.861 ± 0.097 |
| false alarms / hour | 6.6 ± 8.5 |

Averaged over 7 leave-one-recording-out folds. A random detector would score approximately 0.017 AUPRC.

### Synthetic data quality

| check | real reference | synthetic | verdict |
|---|---|---|---|
| spectral match (1–40 Hz) | — | 0.450 | correct shape, magnitudes off |
| nearest-neighbour correlation | 0.212 | 0.232 | new samples, same family |
| inter-sample diversity | 0.000 | 0.102 | comparable to real data |

Two of three checks pass against measured references. The spectral gap is the honest weakness: synthetic windows carry less high-frequency power than real seizures. This plateaued around 0.42–0.45 across every configuration I tried, including different upsampling modes and training lengths, which suggests an architectural limit rather than an untuned hyperparameter.

### Augmentation experiment

| fold | held out | baseline AUPRC | augmented AUPRC | change |
|---|---|---|---|---|
| 1 | chb01_03 | 0.899 | 0.850 | −0.050 |
| 2 | chb01_04 | 0.946 | 0.961 | +0.015 |
| 3 | chb01_15 | 0.692 | 0.742 | +0.050 |
| 4 | chb01_16 | 0.931 | 0.939 | +0.008 |
| 5 | chb01_18 | 0.984 | 0.994 | +0.009 |
| 6 | chb01_21 | 0.979 | 0.981 | +0.002 |
| 7 | chb01_26 | 0.980 | 0.978 | −0.003 |

| | AUPRC |
|---|---|
| real data only | 0.916 ± 0.104 |
| real + 500 synthetic seizure windows | 0.920 ± 0.092 |
| **paired difference** | **+0.004 ± 0.029** |

Five of seven folds improved, but the mean change is roughly seven times smaller than the fold-to-fold variation. **There is no detectable effect.**

Each fold trains two detectors from scratch with the same seed and evaluates both on the same held-out recording. A paired design is necessary here because the baseline itself varies by around ±0.10 across folds; an unpaired comparison would lose a small real effect inside that variation. Synthetic windows are tagged with a placeholder recording name so they can never be selected into a calibration or test fold.

## Key Findings

**Synthetic data did not improve detection.** With 230 seizure windows drawn from 7 seizures in a single patient, a generator can only recombine variation that already exists. It cannot introduce seizure morphologies the patient never had. Adding 500 recombinations of material the detector had already seen gave it nothing new to learn from.

**Augmentation shifted calibration, not capability.** This is the finding I did not expect. Recall dropped sharply in several folds — from 0.714 to 0.238 in one case — while AUPRC held flat and false alarms fell alongside it. The model's ability to rank seizure windows above background was essentially unchanged; what moved was where the calibrated threshold landed. Reported with recall as the headline metric, this run would have read as synthetic data destroying the detector. That conclusion would have been wrong, and the data would have supported it.

**The evaluation protocol was harder than the modelling.** Three separate issues produced misleading results before I had a number worth reporting: leakage from overlapping windows in a random split, epoch selection on the test set, and BatchNorm making the decision threshold untransferable between recordings. The first version of this project reported a perfect F1 of 1.000.

## Limitations

**One patient.** Everything here comes from `chb01`. Nothing in these results speaks to cross-patient generalisation, which is both the harder problem and the clinically relevant one.

**The generator's spectrum is imperfect.** Synthetic windows contain less high-frequency content than real seizures. This means the negative result is not fully clean: I cannot completely separate "augmentation does not help at this sample size" from "this particular generator was not good enough."

**Artefacts are unhandled.** Raw amplitudes reach ±1861 µV, far outside physiological EEG at roughly ±200 µV — electrode pops and movement. Per-window standardisation limits the damage, but a window containing a large artefact is normalised by that artefact, flattening the real signal. This is the most likely explanation for the weakest fold (`chb01_15`, AUPRC 0.692).

**A single augmentation dose.** Only 500 synthetic windows were tested. A dose-response sweep would make the null result considerably harder to argue with.

**Device-dependent numbers.** The baseline measured on CPU (0.898 ± 0.138) differs from the same protocol on Apple Silicon GPU (0.916 ± 0.104) due to floating-point differences between backends; one fold moved from 0.566 to 0.692. The augmentation comparison is unaffected since both arms run on the same device, but baseline figures should be quoted with their device.

## Future Work

1. **Artefact rejection or amplitude clipping** before windowing, then re-establish the baseline. This is the cheapest change and would likely raise the weakest fold.
2. **Dose-response sweep** at 250, 1000 and 2000 synthetic windows, to close the obvious objection that only one amount was tested.
3. **Extend to multiple patients.** This is the change most likely to alter the conclusion, since it gives the generator genuinely new seizure morphologies rather than recombinations of seven.

## How to Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the scripts in order:

```bash
python scripts/01_check_setup.py
python scripts/02_download_chbmit.py       # ~340 MB from PhysioNet
python scripts/03_parse_summary.py
python scripts/04_make_windows.py
python scripts/05_train_detector.py        # baseline cross-validation
python scripts/06_visualize_windows.py
python scripts/07_train_gan.py
python scripts/08_inspect_synthetic.py
python scripts/09_augment_and_evaluate.py  # the experiment
```

Each script checks that its inputs exist and names the script to run if they do not, so the dependency chain is self-documenting. Device selection is automatic, preferring CUDA, then Apple Silicon GPU, then CPU.

Useful flags:

```bash
python scripts/07_train_gan.py --quick                    # 3-epoch sanity check
python scripts/05_train_detector.py --norm batch          # reproduce the BatchNorm behaviour
python scripts/09_augment_and_evaluate.py --n-synthetic 1000
```

## Main Takeaway

The detector works: roughly 0.90 AUPRC on data where a random model scores 0.017.

The augmentation question came back negative, and I think the negative answer is the more useful contribution. A generator trained on seven seizures from one patient can only reshuffle what is already there, and a properly controlled experiment shows that reshuffling does not add information the detector did not have.

The part I would emphasise is not the result but the path to it. The first version of this project reported a perfect F1 of 1.000, and it took finding three separate evaluation problems to reach a number I would defend. Most of the work here is in building an experiment whose answer can be trusted — the answer itself is just what that experiment returned.

## Data Source

Shoeb, A. *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment.* PhD thesis, MIT, 2009. CHB-MIT Scalp EEG Database, PhysioNet.
