from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from morpheus_scorekit.freefall_sweep import (
    evaluate_sweep,
    load_manifest,
    render_sweep_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and report a generated free-fall gravity dose sweep."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--threshold", type=float, help="Omit to calibrate from normal data")
    parser.add_argument("--minimum-auto-threshold", type=float, default=0.02)
    parser.add_argument("--gravity", type=float, default=-9.81)
    args = parser.parse_args()

    output = args.output_dir or args.manifest.parent / "freefall_sweep_report"
    output.mkdir(parents=True, exist_ok=True)
    result = evaluate_sweep(
        load_manifest(args.manifest),
        expected_gravity=args.gravity,
        window_size=args.window_size,
        threshold=args.threshold,
        minimum_auto_threshold=args.minimum_auto_threshold,
    )
    json_path = output / "sweep_results.json"
    report_path = output / "report.md"
    csv_path = output / "per_video_results.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(render_sweep_markdown(result), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps({"report": str(report_path), "json": str(json_path), "csv": str(csv_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
