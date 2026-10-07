from __future__ import annotations

import argparse
import json
from pathlib import Path

from morpheus_scorekit.freefall_diagnostics import compare_freefall_pair, render_pair_markdown


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare a Kubric normal/violation pair using actual time and world coordinates."
    )
    parser.add_argument("pair_dir", type=Path)
    parser.add_argument(
        "--score-root",
        type=Path,
        help="Directory containing normal_gt/ and violation_gt/ Morpheus results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: PAIR_DIR/morpheus/freefall_diagnostics",
    )
    parser.add_argument("--gravity", type=float, default=-9.81)
    args = parser.parse_args()

    output_dir = args.output_dir or args.pair_dir / "morpheus" / "freefall_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = compare_freefall_pair(
        args.pair_dir, score_root=args.score_root, expected_gravity=args.gravity
    )
    json_path = output_dir / "diagnostics.json"
    report_path = output_dir / "report.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(render_pair_markdown(result), encoding="utf-8")
    print(json.dumps({"diagnostics": str(json_path), "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
