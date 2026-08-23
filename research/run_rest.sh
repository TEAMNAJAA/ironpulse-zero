#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
LOG=results/run_rest.log
: > "$LOG"

run () {
  echo "" | tee -a "$LOG"
  echo "################ $1 ################" | tee -a "$LOG"
  date -Is | tee -a "$LOG"
  python "$2" 2>&1 | grep -vE "^  (build|multi) " | tee -a "$LOG"
  echo "---- finished $1" | tee -a "$LOG"
}

run "E5 transfer"          experiments/e5_transfer.py
run "E3 detectors"         experiments/e3_detectors.py
run "E4 features"          experiments/e4_features.py
run "E6 noise"             experiments/e6_noise.py
run "E8 trend"             experiments/e8_trend.py
run "spectra figures"      experiments/fig_spectra.py
run "figures"              experiments/make_figures.py
run "summary"              experiments/summarize.py

echo "" | tee -a "$LOG"
echo "ALL REMAINING EXPERIMENTS COMPLETE" | tee -a "$LOG"
date -Is | tee -a "$LOG"
