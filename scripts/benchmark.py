#!/usr/bin/env python3
"""Run a reproducible PhasePhyto-vs-baseline domain-shift benchmark.

The script orchestrates the existing training and evaluation CLIs, audits class
overlap first, and writes a compact Markdown/JSON summary.  It does not invent
metrics; it records exactly what the evaluation entry points produce.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_class_overlap import audit_overlap  # noqa: E402


def run_command(cmd: list[str]) -> None:
    """Run a subprocess command and stream output.

    Args:
        cmd: Command vector to execute.
    """
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON object.
    """
    with path.open() as f:
        return json.load(f)


def metric_row(name: str, results: dict[str, Any]) -> str:
    """Format one Markdown benchmark row.

    Args:
        name: Model display name.
        results: Evaluation results dictionary.

    Returns:
        Markdown table row.
    """
    source = results["source"]
    target = results["target"]
    delta = results["delta"]
    return (
        f"| {name} | {source['accuracy']:.4f} | {target['accuracy']:.4f} | "
        f"{delta['accuracy_drop']:+.4f} | {source['f1_macro']:.4f} | "
        f"{target['f1_macro']:.4f} | {delta['f1_drop']:+.4f} |"
    )


def write_summary(
    output_dir: Path,
    audit: dict[str, Any],
    phasephyto_results: dict[str, Any],
    baseline_results: dict[str, Any],
) -> None:
    """Write benchmark JSON and Markdown summary artifacts.

    Args:
        output_dir: Directory for generated benchmark artifacts.
        audit: Class-overlap audit report.
        phasephyto_results: PhasePhyto evaluation results.
        baseline_results: Baseline evaluation results.
    """
    summary = {
        "audit": audit,
        "phasephyto": phasephyto_results,
        "baseline": baseline_results,
    }
    (output_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    markdown = [
        "# PhasePhyto Benchmark Summary",
        "",
        f"- Source root: `{audit['source_root']}`",
        f"- Target root: `{audit['target_root']}`",
        f"- Overlapping normalized classes: {audit['overlap_num_classes']}",
        "",
        "| Model | Source Acc | Target Acc | Acc Delta | Source F1 | Target F1 | F1 Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
        metric_row("PhasePhyto", phasephyto_results),
        metric_row("Semantic baseline", baseline_results),
        "",
    ]
    (output_dir / "benchmark_summary.md").write_text("\n".join(markdown))


def main() -> None:
    """Run the benchmark orchestration CLI."""
    parser = argparse.ArgumentParser(description="Benchmark PhasePhyto against baseline")
    parser.add_argument("--config", default="configs/plant_disease.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--source-dir", type=Path, help="Override source eval root")
    parser.add_argument("--target-dir", type=Path, help="Override target eval root")
    parser.add_argument("--phasephyto-ckpt", type=Path, help="Existing PhasePhyto checkpoint")
    parser.add_argument("--baseline-ckpt", type=Path, help="Existing baseline checkpoint")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only evaluate existing checkpoints",
    )
    parser.add_argument("--epochs", type=int, help="Optional training epoch override")
    parser.add_argument("--batch-size", type=int, help="Optional training batch-size override")
    parser.add_argument("--device", help="Optional device override, e.g. cpu or cuda")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source = args.source_dir or Path("data/plant_disease/plantvillage")
    target = args.target_dir or Path("data/plant_disease/plantdoc")
    audit = audit_overlap(source, target)
    (output_dir / "class_overlap.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["overlap_num_classes"] == 0:
        raise SystemExit("No class overlap; refusing to benchmark.")

    overrides: list[str] = []
    if args.epochs is not None:
        overrides.append(f"training.epochs={args.epochs}")
    if args.batch_size is not None:
        overrides.append(f"training.batch_size={args.batch_size}")
    if args.device is not None:
        overrides.append(f"device={args.device!r}")

    if not args.skip_train:
        train_cmd = [sys.executable, "-m", "phasephyto.train", "--config", args.config]
        baseline_cmd = [
            sys.executable,
            "-m",
            "phasephyto.train_baseline",
            "--config",
            args.config,
        ]
        if overrides:
            train_cmd.extend(["--override", *overrides])
            baseline_cmd.extend(["--override", *overrides])
        run_command(train_cmd)
        run_command(baseline_cmd)

    phasephyto_ckpt = args.phasephyto_ckpt or Path("checkpoints/plant_disease/best_model.pt")
    baseline_ckpt = args.baseline_ckpt or Path("checkpoints/plant_disease/baseline/best_model.pt")
    phasephyto_json = output_dir / "phasephyto_eval.json"
    baseline_json = output_dir / "baseline_eval.json"

    run_command([
        sys.executable,
        "-m",
        "phasephyto.evaluate",
        "--config",
        args.config,
        "--checkpoint",
        str(phasephyto_ckpt),
        "--source-dir",
        str(source),
        "--target-dir",
        str(target),
        "--output",
        str(phasephyto_json),
    ])
    run_command([
        sys.executable,
        "-m",
        "phasephyto.evaluate_baseline",
        "--config",
        args.config,
        "--checkpoint",
        str(baseline_ckpt),
        "--source-dir",
        str(source),
        "--target-dir",
        str(target),
        "--output",
        str(baseline_json),
    ])

    write_summary(output_dir, audit, load_json(phasephyto_json), load_json(baseline_json))
    print(f"\nBenchmark artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
