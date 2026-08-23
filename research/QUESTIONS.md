# QUESTIONS.md — items needing the user's decision

Work continues past every item here; nothing below is blocking.

## Q1 — High-pass cutoff for the acceleration-to-displacement conversion (RESOLVED, deviation logged)

Spec 6.2.1 says to high-pass at about 0.5 Hz. Applied literally, the 1/f^2 integration gain
(3600x between 0.5 Hz and 30 Hz) leaves a sub-Hz artifact **10x larger than the true signal**:
on a synthetic 50 um / 30 Hz reference, the literal filter returns a 521.4 um peak at 0.5 Hz and a
1352 um record maximum.

Adopted instead: effective cutoff `max(0.5 Hz, 0.1 * f0)`. At f0 = 30 Hz this is 3 Hz, which is
exactly the lower edge of the order-analysis band (order 0.1), so nothing inside the analysed band
is touched. It recovers the reference as 50.00000 um vs 50.00000 um ground truth. The cutoff scales
with speed, which also keeps the whole chain order-consistent for the E5 cross-speed transfer test.

**No action needed unless you disagree.** Set `integration.highpass_order_floor: 0` in `config.yaml`
to revert to the literal reading.

## Q2 — Which vibration channel represents "what the camera sees"

MAFAULDA has 8 channels. Primary channel used is **underhang radial** (column 3, index 2), because
a camera pointed at the machine sees radial motion in the image plane. Axial motion is along the
optical axis and largely invisible to a single camera. Cross-channel comparison is not part of the
spec's experiment list; say the word and it can be added.

## Q3 — MAFAULDA is 12.9 GB, under the 20 GB gate, so it is being downloaded in full

No stratified subset was needed for the download. A stratified subset may still be used for
processing time; if so it will be recorded in `datasets/subset_manifest.csv` as the spec requires.

## Q4 — E4 contradicts the spec's ban on raw-unit features (DECISION NEEDED)

Spec 6.3 says to keep the raw values for plotting but **never feed them to the model**, and E4 was
framed as proving that dimensionless normalisation is necessary. **The measurement says the
opposite.** At 240 Hz, boxcar, displacement, Mahalanobis:

| feature set | MAFAULDA | CWRU |
|---|---|---|
| 13 dimensionless (spec) | 0.7147 | 0.6922 |
| dimensionless + raw amplitude | **0.7656** | **0.8513** |

That is +0.051 on MAFAULDA and **+0.159 on CWRU**. It is consistent with E8, where the spec
feature set could not do prognostics at all because degradation shows up in amplitude rather than
in spectral shape.

The counter-argument is transferability: raw amplitude is in absolute pixels and therefore moves
with camera distance, camera angle and shaft speed. E4b tests that directly (cross-speed transfer
with and without raw features) and its result is in FINDINGS 2.6c.

**Question for you:** if E4b shows raw features do not break cross-speed transfer, do you want the
web app's feature set changed to dimensionless + raw amplitude? That is a deviation from the spec,
so it is your call, not mine. My recommendation either way is that **at least one absolute
amplitude feature is mandatory if the project is going to use the words "predictive maintenance"**
(E8: 74 h of warning with amplitude vs no warning at all without it).

Note that raw features force a stricter capture discipline: identical camera distance, angle and
framing every session, with the tripod position marked on the floor.

## Q5 — Tooling note

7-Zip (already installed on this machine) was used to unpack the `.rar` archives nested inside
NASA's `IMS.zip`. It is outside the dependency list in spec section 3, but it is used only for
extraction, never for analysis. No Python package outside the spec list was added; `librosa` was
installed because it is on the spec list, though nothing in the final pipeline imports it.

### Q4 — ANSWERED by E4b, but one decision is still yours

E4b tested the transferability counter-argument directly. Result: **the spec's ban is half right.**

- When test speed matches training speed, adding raw amplitude is clearly better: **+0.063** on
  MAFAULDA (0.715 -> 0.778) and **+0.045** on CWRU (0.692 -> 0.737).
- When the speed genuinely changes, adding raw amplitude **hurts**: MAFAULDA slow->fast falls from
  0.798 to 0.721, and the false-alarm rate in the reverse direction jumps from 0.574 to **0.892**.
- CWRU cannot settle this question: its speed range is only 4 % wide (1730-1797 rpm), so raw
  amplitudes barely move. Only MAFAULDA, with a 5:1 speed range, can test it.

Mechanism, as predicted: imbalance amplitude scales with omega^2, so raw pixel amplitudes change by
a large factor across a 5:1 speed range while the dimensionless ratios do not.

**Recommended resolution** — pick based on your machine, not on the spec:

1. Constant-speed machine (most fans and pumps): use **dimensionless + raw**. It is better.
2. Widely varying speed: dimensionless only for detection, and carry absolute amplitude as a
   **separate trend channel** for prognostics (E8 shows prognostics is impossible without it).
3. Never raw-only in any case — worst cross-speed transfer of the three (0.588).

The remaining decision is only whether you want the web app changed. Nothing is blocked either way.
