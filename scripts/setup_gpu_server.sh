#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Set SAM2_BUILD_CUDA=0 before running this script if the server cannot compile
# the optional CUDA extension. PyTorch CUDA inference still works without it.
: "${SAM2_BUILD_CUDA:=1}"
export SAM2_BUILD_CUDA

python -m pip install -e ".[sam2]"
python -m pip install -e vendor/Morpheus/sam2
python -m pip install -e vendor/Morpheus

CHECKPOINT_DIR="$ROOT/vendor/Morpheus/checkpoints"
CHECKPOINT="$CHECKPOINT_DIR/sam2.1_hiera_large.pt"
mkdir -p "$CHECKPOINT_DIR"
if [[ ! -f "$CHECKPOINT" ]]; then
    URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    if command -v wget >/dev/null 2>&1; then
        wget -O "$CHECKPOINT" "$URL"
    elif command -v curl >/dev/null 2>&1; then
        curl -L "$URL" -o "$CHECKPOINT"
    else
        echo "wget or curl is required to download the SAM2 checkpoint" >&2
        exit 1
    fi
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m morpheus_scorekit.cli sam2-doctor --device cuda

echo
echo "Ready. Activate later with: source $ROOT/.venv/bin/activate"
