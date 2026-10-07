#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAIR_DIR="${1:?usage: run_kubric_pair.sh PAIR_DIR [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-$PAIR_DIR/morpheus}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-10000}"
FPS="${FPS:-24}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PAIR_DIR="$(cd "$PAIR_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

for split in normal violation; do
  trajectory="$PAIR_DIR/$split/trajectory_freefall_gt.npz"
  if [[ ! -f "$trajectory" ]]; then
    echo "Missing $trajectory. Regenerate this pair with the current kubric-physics-dataset." >&2
    exit 2
  fi
  "$PYTHON_BIN" -m morpheus_scorekit.cli score \
    "$trajectory" \
    --experiment falling_ball \
    --output-dir "$OUTPUT_DIR/${split}_gt" \
    --generated \
    --epochs "$EPOCHS"
done

"$PYTHON_BIN" "$ROOT/scripts/compare_freefall_pair.py" \
  "$PAIR_DIR" \
  --score-root "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR/freefall_diagnostics"

"$PYTHON_BIN" "$ROOT/scripts/prepare_kubric_freefall_frames.py" \
  "$PAIR_DIR" "$OUTPUT_DIR/freefall_frames"

for split in normal violation; do
  "$PYTHON_BIN" -m morpheus_scorekit.cli run-sam2 \
    "$OUTPUT_DIR/freefall_frames/$split" \
    --auto-seed \
    --seed-preview "$OUTPUT_DIR/${split}_video/seed.jpg" \
    --fps "$FPS" \
    --experiment falling_ball \
    --output-dir "$OUTPUT_DIR/${split}_video" \
    --device "$DEVICE" \
    --generated \
    --epochs "$EPOCHS"
done

echo "GT scores:    $OUTPUT_DIR/normal_gt and $OUTPUT_DIR/violation_gt"
echo "Video scores: $OUTPUT_DIR/normal_video and $OUTPUT_DIR/violation_video"
echo "Diagnostics:  $OUTPUT_DIR/freefall_diagnostics/report.md"
