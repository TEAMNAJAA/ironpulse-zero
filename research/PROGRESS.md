# IronPulse Zero — Research Progress Log

Started: 2026-08-20
Working dir: `C:\Users\more_\Downloads\Arise\research`

---

## HEADLINE FINDINGS

**1. No public 240 fps labelled video dataset of rotating machinery exists.** Searched within the
20-minute budget; only self-constructed per-paper footage. The accelerometer down-sampling plan in
spec section 1 is therefore the correct route. Written up in `datasets/SURVEY.md`.

**2. The biggest loss is the measurement DOMAIN, not the frame rate.** An accelerometer measures
acceleration; a camera measures displacement. At an identical 48 kHz rate, converting to
displacement drops CWRU from **AUC 0.9997 to 0.750** and MAFAULDA from **0.906 to 0.820**.
Cause, measured: in acceleration, 65-99 % of CWRU bearing-fault energy sits at 1-6 kHz; double
integration weights that band by 1/f^2 and collapses ~100 % of the displacement energy below
200 Hz. Unavoidable for **any** camera at **any** frame rate.

**3. E1 self-check gate PASSED.** Spec section 7 requires AUC >= 0.85 on CWRU. With the exact spec
feature set in the acceleration domain: **AUC = 0.9997 +/- 0.0007** (4-fold, grouped by file).
The lower displacement number is physics, not a bug — see `results/tables/diag_domain_cwru.csv`.

**4. WITHOUT ANTI-ALIASING A 240 fps CAMERA RENAMES FAULTS RATHER THAN MISSING THEM.**
Nyquist at 240 fps is a fixed 120 Hz, so a defect line above it folds back onto a *lower* order that
often coincides with a real signature. Computed from the manufacturers' own bearing geometry
(`results/tables/aliasing_analysis.csv`):

  | Defect | Shaft speed | True order | Reported order | Masquerades as |
  |---|---|---|---|---|
  | MAFAULDA BPFI inner race | 2400 rpm | 5.002x | **0.998x** | 1x imbalance, gap 0.002 |
  | MAFAULDA BPFO outer race | 3510 rpm | 2.998x | 1.105x | 1x imbalance |
  | MAFAULDA BPFO outer race | 3000 rpm | 2.998x | 1.802x | 2x misalignment |
  | CWRU BSF ball | 1794 rpm | 4.714x | 3.313x | 3x misalignment |
  | CWRU BPFI inner race | 1795 rpm | 5.415x | 2.612x | near 2x/3x |

  **Mitigation is counter-intuitive: use the SLOWEST shutter the frame rate allows, not the
  fastest.** Exposure averaging is a sinc-shaped anti-alias filter. At 240 fps a 1/240 s exposure
  cuts that BPFI alias to 19 % of its amplitude; 1/1000 s leaves ~96 % of it. E2c quantifies this.

**5. Speed, not frame rate, is the binding constraint.** Usable order range at 240 fps = 120/f0:

  | Component | Order | Highest shaft speed still resolved |
  |---|---|---|
  | 1x imbalance | 1.000x | 7200 rpm |
  | 2x misalignment | 2.000x | 3600 rpm |
  | 3x misalignment | 3.000x | **2400 rpm** |
  | MAFAULDA BPFO | 2.998x | 2402 rpm |
  | MAFAULDA BPFI | 5.002x | **1439 rpm** |
  | CWRU BPFI | 5.415x | 1330 rpm |

**6. A confound in MAFAULDA's bearing tests, found and controlled.** MAFAULDA's bearing-fault runs
carry 0/6/20/35 g of *added rotor mass*, so "bearing fault detected" can really be "added imbalance
detected". Re-running restricted to the pure 0 g defects (`results/tables/e1b_confound.csv`) changes
the conclusion materially — at 240 Hz boxcar, displacement:

  | Class | all severities | pure defect (0 g) |
  |---|---|---|
  | underhang outer race | 0.702 | **0.549** (chance) |
  | underhang cage | 0.706 | **0.575** (chance) |
  | underhang ball | 0.653 | **0.554** (chance) |
  | overhang cage | 0.793 | **0.562** (chance) |
  | overhang outer race | 0.888 | 0.865 |
  | overhang ball | 0.878 | 0.826 |

  So most of the apparent bearing-fault performance at 240 Hz is the added mass, not the defect.
  Only the **overhang** position (bearing outboard of the rotor, long lever arm to the shaft orbit)
  keeps a genuine displacement signature. Reported honestly rather than quoting the flattering
  all-severities number.

**7. Misalignment is the hardest class at every rate and in both domains** — MAFAULDA horizontal
misalignment scores 0.722 in acceleration, 0.619 in displacement at full rate, and **0.507 (pure
chance) at 240 Hz**. Vertical misalignment: 0.743 / 0.692 / 0.559. This is partly the rig (offsets
of only 0.5-2.0 mm) and partly the 2x/3x lines dropping past Nyquist above 3600/2400 rpm.

**8. E8 decomposes the prognostics claim, and it replicates on a second failure.**
IMS run-to-failure, test 2 (163.5 h) and test 3 (1073.1 h):

  | Domain | Features | test 2 | test 3 |
  |---|---|---|---|
  | acceleration, full rate | spec dimensionless | **74.0 h** | **49.7 h** |
  | acceleration, full rate | + absolute amplitude | 74.7 h | 59.0 h |
  | displacement, 48 kHz / 240 Hz | spec dimensionless | **never alarms** | **never alarms** |
  | displacement, 240 Hz | + absolute amplitude | **2.2 h** | **5.0 h** |

  Frame rate costs nothing here (240 Hz equals 48 kHz exactly). The **spec's dimensionless feature
  set cannot do prognostics at all**: degradation appears as *amplitude* (displacement RMS
  0.032 -> 0.072 px) while every ratio stays flat. **Recommendation: the web app must keep at least
  one absolute-amplitude feature to claim predictive maintenance.**

**9. A necessary deviation from the spec's 0.5 Hz high-pass.** Applied literally it leaves a
1/f^2-amplified sub-Hz artifact **10x larger than the true signal** (521.4 um artifact vs 50 um ground
truth). Effective cutoff used: `max(0.5 Hz, 0.1*f0)`, the lower edge of the order-analysis band,
which recovers the reference as **50.00000 um vs 50.00000 um**. Logged as Q1 in `QUESTIONS.md`; one
config key reverts it.

**10. Scale reality check.** A 50 um vibration at 30 cm on 720p (29 px/cm) spans **0.145 px**. In
the IMS run, healthy displacement RMS is 0.032 px and only reaches 0.072 px at failure. Sub-pixel
optical flow is mandatory; E7 pins down how sub-pixel.

---

## Status

| Phase | Task | Status |
|---|---|---|
| 0 | Scaffold, env check | DONE |
| 1 | `datasets/SURVEY.md` | DONE |
| 2 | CWRU 161 files, 688 MB, 0 failures | DONE |
| 2 | MAFAULDA 1951 files, 12.9 GB, counts match docs exactly | DONE |
| 2 | IMS 1.06 GB + tests 2 and 3 extracted | DONE |
| 3 | `core/` pipeline | DONE |
| 4 | E1 baseline both domains both datasets | DONE |
| 4 | E1b bearing-fault confound control | DONE |
| 4 | Aliasing / Nyquist analysis (+2 figures) | DONE |
| 4 | E8 trend on IMS tests 2 and 3 | DONE |
| 4 | E2 sampling rate, all 1951 files | DONE |
| 4 | E2b speed band | DONE |
| 4 | E2c exposure sweep (1/240, 1/500, 1/1000 s) | DONE |
| 4 | E7 resolution + E7b amplitude census | DONE |
| 4 | E5, E3, E4, E4b, E6, figures | DONE |
| 5 | `CITATIONS.md`, `subset_manifest.csv` (9420 rows) | DONE |
| 5 | `FINDINGS.md` (Thai), `POSTER_NUMBERS.md` | DONE, all sections filled |

**Note on the interrupted run:** the first `run_all.sh` was killed by a process teardown partway
through E5. Everything up to and including E7 had already written its CSV, so nothing was lost.
E5 onwards was re-run by `run_rest.sh`, launched detached with `nohup`, and completed at 08:20.

## Environment

- Python 3.11.9; numpy scipy scikit-learn pandas matplotlib requests tqdm soundfile pyyaml librosa
- Disk free ~160 GB (floor 10 GB). No GPU, no deep-learning framework — per spec section 3
- 7-Zip (pre-installed) used only to unpack the `.rar` files inside `IMS.zip`, never for analysis

## Data integrity

| Dataset | Files | Size | Integrity |
|---|---|---|---|
| CWRU | 161 / 161 | 688 MB | sha256 for all 161 in `cwru_download_log.csv`; 4 needed a resumed retry |
| MAFAULDA | 1951 / 1951 | 12.9 GB | sha256 per archive; per-class counts match the published table exactly |
| IMS | 3 tests | 1.06 GB | sha256 `6cb42c26...` |

Two packaging traps found in IMS and written up in `SURVEY.md`: `3rd_test.rar` extracts to a folder
named `4th_test/txt`, and the readme understates test 3 — it documents 4,448 files to
2004-04-04, but the archive holds 6,324 files to 2004-04-18 and **the failure is ~320 h past the
documented end date**. Trimming test 3 to its documented dates analyses a run that never fails.

## Method guarantees (spec section 10)

- Windows are grouped by source file in every split; no file appears on both sides of a fold.
- Only normal data ever trains a detector; anomalous files are test-only by construction.
- Thresholds are fixed at the 99th percentile of training scores before any test data is seen, and
  never retuned.
- Every random draw is seeded from `config.yaml: seed: 42`.
- CWRU has only 4 normal recordings (one per motor load), so the fold count adapts to 4 there. Its
  acceleration-domain false-alarm rate is consequently high (0.49) even at AUC 0.9997: each fold
  trains on 3 loads and tests on an unseen load, so the ranking is perfect but the absolute
  threshold does not transfer. MAFAULDA, with 49 normal files, gives FAR 0.02. Reported, not hidden.
- MAFAULDA f0 comes from the tachometer channel, not the filename — the filename is a nominal
  setpoint running ~2 % high and the tach pulse train has strong harmonics, so the peak is searched
  within +/-20 % of nominal and parabolically interpolated.
- Everything is resampled to a common 48 kHz base so all seven target rates are exact integer
  decimations (MAFAULDA's native 50 kHz would give a non-integer factor at 480 and 240 Hz).

## Camera model validated against closed-form physics

- double integration recovers a 50.00000 um reference as **50.00000 um**
- boxcar decimation of 30 Hz at 240 fps, 1/240 s exposure gives **48.72477 um** against the
  theoretical `50 * sinc(30/240) = 48.72477 um` — exact to every digit shown
- naive decimation gives exactly **50.000 um** (no attenuation, but it aliases)
- all four detectors verified on separable synthetic data (AUC 0.996-1.000)

## Results confirmed since the last update

- **E2 (main poster graph).** MAFAULDA displacement, 1951 files: full 48 kHz = 0.8195; at 240 Hz
  boxcar = **0.7147, a 12.8 % drop** (ideal 0.6946, naive 0.6973). CWRU at 240 Hz boxcar 0.6922,
  a 7.7 % drop. **The curve is flat from 48 kHz down to 1 kHz** — sampling faster than 1 kHz buys a
  displacement sensor nothing. Degradation starts at 480 Hz.
  The three modes are **not** significantly different for detection (spread 0.695–0.715 against
  s.d. 0.059–0.071); they differ in stability and in *which order* the energy lands on.
- **E2b.** A hypothesis of mine failed: AUC does *not* fall monotonically with shaft speed. The
  10–20 Hz band is worst at **every** rate including full 48 kHz (0.577), so that is 1/f²
  integration noise, not Nyquist. Isolating the rate effect, the worst loss is the 30–40 Hz band
  (−0.225). Corrected in FINDINGS §3.2b.
- **E2c.** The slow-shutter prediction held: the longest exposure wins in **5 of 6** rate × dataset
  combinations, biggest effect CWRU at 240 Hz (**+0.047** going from 1/1000 s to 1/240 s).
  Effect size is below the per-point s.d., so the strength is in the consistent direction, which
  was predicted from the sinc analysis beforehand rather than found after the fact.
- **E7 + E7b.** MAFAULDA is unaffected even at a coarse 0.1 px step (0.715 → 0.729); CWRU is
  destroyed (0.692 → 0.496, chance). The governing quantity is the discriminative line amplitude
  against quantisation noise q/√12: every MAFAULDA class sits at ratio ≥ 4 at 0.05 px, CWRU at 0.02.
  **Rule: q ≤ 0.85 × the weakest line you need. At 0.05 px that means the 1x vibration must
  exceed 21 µm.**

- **E5 (starred).** The order axis is vindicated, but **the evidence is the false-alarm rate, not
  the AUC**. Training slow and testing fast: order axis 0.798 at FAR **0.017**; fixed-Hz axis 0.669
  at FAR **1.000** — it flags every healthy window. CWRU cross-load: 0.733 / FAR 0.055 vs
  0.413 / FAR 0.945. Two deployment rules follow: transfer is **asymmetric** (fast→slow gives 0.436,
  below chance, FAR 0.574 — train at or below the deployment speed), and **cross-machine transfer
  fails outright** (MAFAULDA→CWRU = 0.504, exactly chance; every machine needs its own baseline).
- **E3.** OneClassSVM 0.785 ± 0.033 beats Mahalanobis 0.715 ± 0.059 at the same 240 Hz —
  **recovering 67 % of the entire frame-rate loss**, at half the variance, in 0.8 s. Every other
  number in the report uses Mahalanobis (third best), so all reported figures **understate** the
  system. That choice was fixed before E3 ran.
- **E4 + E4b — a result that contradicts the spec.** Spec 6.3 forbids feeding raw-unit features to
  the model and framed E4 as proving dimensionless normalisation necessary. Measured: adding raw
  amplitude gives **+0.051 on MAFAULDA and +0.159 on CWRU**. E4b then tested the transferability
  counter-argument directly and found the ban is **half right**: with matched speeds raw features
  help (+0.063), but across a genuine 5:1 speed range they hurt (0.798 → 0.721, and FAR 0.574 →
  0.892), because imbalance amplitude scales with ω². CWRU cannot settle it — its speed range is
  only 4 % wide. Logged as Q4 in `QUESTIONS.md` with a recommended resolution; the decision on
  whether to change the web app is the user's.
  Also from E4: `rbp` is inert here (neither dataset has blades — dropping it changes nothing, and
  alone it scores exactly 0.5000), and crest+kurtosis alone on CWRU score **0.438, below chance** —
  kurtosis, the classic bearing indicator in acceleration, inverts in the displacement domain.

- **E6.** Camera shake is almost harmless: at 0 dB SNR (shake as strong as the whole signal)
  MAFAULDA loses only **3.9 %** (0.715 -> 0.687) and CWRU essentially nothing (0.692 -> 0.683),
  because shake is narrowband 8-12 Hz and corrupts only `E_sub`. Broadband white noise is the real
  enemy: CWRU falls to 0.596 at 30 dB and **0.480 (below chance) at 0 dB**. Practical consequence:
  an ordinary tripod is fine; spend the effort on optical-flow precision and machine choice.
- **E8 repeated on a second, independent failure (IMS test 3, 1073 h).** Same conclusions:
  acceleration 49.7 h lead (74.0 h on test 2); displacement at 240 Hz with amplitude 5.0 h
  (2.2 h on test 2); **the spec's dimensionless set never alarms, in either failure.** n = 2 is
  still too few for confidence intervals - the numbers are an order of magnitude, not an estimate.

## Verification

A cross-check script re-derived **25 headline numbers** in `FINDINGS.md` and `POSTER_NUMBERS.md`
directly from the CSVs in `results/tables/`. **All 25 matched; no mismatches.** Nothing in either
report is transcribed by hand from a log.

## Final deliverables

| File | State |
|---|---|
| `datasets/SURVEY.md` | 7 datasets, every URL fetched live, 2 packaging traps in IMS documented |
| `datasets/subset_manifest.csv` | 9,420 rows: every file used, with f0, f0 method and selection reason |
| `datasets/cwru_manifest.csv`, `*_download_log.csv` | sha256 for all downloads |
| `FINDINGS.md` | Thai, all six sections required by spec 8, E1-E8 complete |
| `POSTER_NUMBERS.md` | Thai, quote-ready sentences with matching figure filenames |
| `CITATIONS.md` | citation text for the 3 datasets used + 4 surveyed |
| `QUESTIONS.md` | 5 items, none blocking; Q4 needs a decision from the user |
| `results/tables/` | 24 CSVs |
| `results/figures/` | 25 PNGs at 300 dpi, colourblind-safe (Okabe-Ito + distinct markers/hatching) |
| `results/summary.json` | machine-readable digest of every experiment |
| `config.yaml` | all constants; no magic numbers in code |
| `run_all.sh`, `run_rest.sh` | reproducible end-to-end |
| `synthetic_selftest/` | 18-check validation of the camera model against closed-form physics, all passing |

- **E2d (added after E3).** Re-ran the whole rate sweep with OneClassSVM as well as Mahalanobis,
  since E3 showed the default model was costing ~0.07 AUC. **Corrected headline: at 240 Hz the
  system reaches 0.785 on MAFAULDA, an 11.0 % drop from its own 0.883 full-rate reference, and
  0.772 on CWRU, a 6.7 % drop.** New finding that the weaker model had hidden: with OneClassSVM
  **the curve is flat all the way down to 480 Hz** (0.8828 / 0.8830 / 0.8834 / 0.8844 / 0.8837 —
  spread under 0.002), so **480 fps costs literally nothing** and degradation begins only at
  240 fps. Reported separately as post-hoc model selection rather than replacing the pre-registered
  Mahalanobis numbers.


## Handover to the web app (H1, H2)

`core/` and `synthetic_selftest/` have **moved up one level** to `Arise/core/` and
`Arise/synthetic_selftest/` so that the research project and the web app share one canonical copy,
per HANDOVER section 0. The package is still named `core`, so every `from core import ...` in this
project is unchanged; only the `sys.path` line in the 25 entry points gained one extra level.

Verified after the move: `synthetic_selftest` returns **18 checks, 0 FAIL** with a
**byte-identical** results file (md5 `2702407fa08646e29a257ab99888b051` before and after), and
re-running `experiments/diag_domain.py` reproduces its previous CSV byte for byte.

Two new experiments were added to fill a gap found during H1 — E4 had only ablated feature
*groups* and never swept the DSP parameters the web app needs:
`experiments/e9_dsp_params.py` (detrend, window, order tolerance, band edges, rms floor,
threshold percentile) and `experiments/e9c_bands_and_orders.py` (band decomposition and the
order-group ablation split by whether 3x survives Nyquist). `core/dsp.py` gained polynomial
detrend as a backward-compatible option; the self-test was re-run and still passes 18/18.

Parameters and provenance for the app: `../ironpulse/HANDOVER_PARAMS.md`.

NEXT ACTION: nothing is blocked. Optional follow-ups, in value order: (1) answer Q4 in
`QUESTIONS.md` on whether the web app should adopt dimensionless+raw features; (2) if you want every
section of the report rebuilt on OneClassSVM rather than only the rate sweep (E2d), the other
experiments would each gain roughly +0.06 to +0.08 AUC; (3) register for the Paderborn dataset
(URL in `SURVEY.md`) if real-wear bearing damage is wanted.
