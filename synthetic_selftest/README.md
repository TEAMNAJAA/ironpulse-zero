# synthetic_selftest — code validation only

**Nothing in this folder is a research result and nothing here is reported as one.**
Spec section 10 requires that synthetic data used to test code be kept in a separate, clearly
labelled folder. This is that folder.

`validate_camera_model.py` checks the camera model in `core/camera_sim.py` against closed-form
physics using synthetic signals with known answers. Run it with:

```
python synthetic_selftest/validate_camera_model.py
```

It writes `selftest_results.csv` and exits non-zero if any check fails.

## What it checks (18 checks, all passing)

| Check | Measured | Expected |
|---|---|---|
| double integration recovers a 50 um reference | 50.00000 um | 50.00000 um |
| boxcar shutter 1/240 s vs `50*sinc(f0*T)` | 48.72477 um | 48.72477 um |
| boxcar shutter 1/500 s | 49.70444 um | 49.70444 um |
| boxcar shutter 1/1000 s | 49.92602 um | 49.92601 um |
| naive decimation preserves amplitude | 49.99997 um | 50.00000 um |
| ideal decimation preserves amplitude | 50.19528 um | 50.00000 um (FIR ripple) |
| naive sampling folds 300 Hz to its alias | 60.00 Hz | 60.00 Hz |
| quantisation noise rms = q/sqrt(12) | 0.01444 px | 0.01443 px |
| white noise injected at 20 / 10 / 0 dB SNR | exact | exact |
| 50 um in pixels at 29 px/cm, 30 cm, 720p | 0.14500 px | 0.14500 px |
| one pixel in micrometres | 344.83 um | 344.83 um |
| all four detectors separate a shifted Gaussian | AUC 1.000 | 1.000 |

## One deliberate expected failure

The row `literal 0.5 Hz high-pass leaves a sub-Hz artifact` is **meant** to fail. It demonstrates
why the deviation logged as Q1 in `QUESTIONS.md` was necessary: applying spec 6.2.1's 0.5 Hz
cutoff literally returns a **521.4 um** artifact peak at 0.5 Hz against a 50 um true signal, i.e.
the artifact is more than 10x the thing being measured. Raising the cutoff to
`max(0.5 Hz, 0.1*f0)` recovers the reference exactly.
