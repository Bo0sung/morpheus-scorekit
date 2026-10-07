from __future__ import annotations

import argparse
import json
from pathlib import Path

from morpheus_scorekit.freefall_diagnostics import (
    load_freefall_state,
    scan_freefall_acceleration,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify one world-space free-fall state without a known event window."
    )
    parser.add_argument("state", type=Path, help="Kubric state.npz")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gravity", type=float, default=-9.81)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--residual-threshold", type=float, default=0.15)
    args = parser.parse_args()

    result = scan_freefall_acceleration(
        load_freefall_state(args.state),
        expected_gravity=args.gravity,
        window_size=args.window_size,
        residual_threshold=args.residual_threshold,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
