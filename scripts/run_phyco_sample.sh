#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMPLE_DIR="${1:?usage: run_phyco_sample.sh SAMPLE_DIR [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-$SAMPLE_DIR/morpheus}"
VIDEO="$SAMPLE_DIR/rgba.mp4"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-10000}"
FPS="${FPS:-24}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "$VIDEO" ]]; then
  echo "Missing $VIDEO. Extract a PhyCo sample first." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m morpheus_scorekit.cli extract-sam2 \
  "$VIDEO" \
  --auto-seed \
  --seed-preview "$OUTPUT_DIR/seed.jpg" \
  --fps "$FPS" \
  --experiment falling_ball \
  --output "$OUTPUT_DIR/trajectory_full.npz" \
  --overlay-dir "$OUTPUT_DIR/tracking_overlay" \
  --device "$DEVICE"

"$PYTHON_BIN" "$ROOT/scripts/trim_freefall_trajectory.py" \
  "$OUTPUT_DIR/trajectory_full.npz" \
  "$OUTPUT_DIR/trajectory_freefall.npz"

"$PYTHON_BIN" -m morpheus_scorekit.cli score \
  "$OUTPUT_DIR/trajectory_freefall.npz" \
  --experiment falling_ball \
  --output-dir "$OUTPUT_DIR/scores" \
  --generated \
  --epochs "$EPOCHS"

echo "Trajectory: $OUTPUT_DIR/trajectory_freefall.npz"
echo "Scores:     $OUTPUT_DIR/scores/combined_scores.json"
