#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -eq 2 || $# -gt 4 ]]; then
    echo "Usage: $0 VIDEO [OUTPUT_DIR [SEED_X SEED_Y]]" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO="$1"
OUTPUT_DIR="${2:-$ROOT/outputs/$(basename "${VIDEO%.*}")}"
mkdir -p "$OUTPUT_DIR"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

SEED_ARGS=(--auto-seed --seed-preview "$OUTPUT_DIR/auto_seed_preview.jpg")
if [[ $# -eq 4 ]]; then
    SEED_ARGS=(--seed 1 "$3" "$4")
fi

python -m morpheus_scorekit.cli extract-sam2 \
    "$VIDEO" \
    "${SEED_ARGS[@]}" \
    --experiment falling_ball \
    --device cuda \
    --output "$OUTPUT_DIR/trajectory.npz" \
    --overlay-dir "$OUTPUT_DIR/tracking_overlay"

echo "Trajectory: $OUTPUT_DIR/trajectory.npz"
echo "Overlay:    $OUTPUT_DIR/tracking_overlay"
