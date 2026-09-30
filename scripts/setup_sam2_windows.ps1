$ErrorActionPreference = "Stop"

# SAM2's CUDA extension is optional for inference. Disabling its build makes the
# editable installation work on ordinary Windows/CPU machines as well.
$env:SAM2_BUILD_CUDA = "0"

python -m pip install -e ".[sam2]"
python -m pip install -e "vendor/Morpheus/sam2"

Write-Host "SAM2 code dependencies are installed."
Write-Host "Place sam2.1_hiera_large.pt in vendor/Morpheus/checkpoints/ or pass --checkpoint."
Write-Host "Then run: python -m morpheus_scorekit.cli sam2-doctor"
