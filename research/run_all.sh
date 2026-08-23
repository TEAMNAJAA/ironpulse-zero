#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
LOG=results/run_all.log
: > "$LOG"

run () {
  echo "" | tee -a "$LOG"
  echo "################ $1 ################" | tee -a "$LOG"
  date -Is | tee -a "$LOG"
  python "$2" 2>&1 | grep -vE "^  (build|multi) " | tee -a "$LOG"
  echo "---- finished $1" | tee -a "$LOG"
}

run "E1 baseline"          experiments/e1_baseline.py
run "E2 sampling rate"     experiments/e2_sampling_rate.py
run "E2b speed band"       experiments/e2b_speed_band.py
run "E7 resolution"        experiments/e7_resolution.py
run "E5 transfer"          experiments/e5_transfer.py
run "E3 detectors"         experiments/e3_detectors.py
run "E4 features"          experiments/e4_features.py
run "E6 noise"             experiments/e6_noise.py
run "E8 trend"             experiments/e8_trend.py
run "spectra figures"      experiments/fig_spectra.py
run "figures"              experiments/make_figures.py

echo "" | tee -a "$LOG"
echo "ALL EXPERIMENTS COMPLETE" | tee -a "$LOG"
date -Is | tee -a "$LOG"
