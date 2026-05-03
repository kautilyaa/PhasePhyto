#!/usr/bin/env python3
"""Prepare lightweight demo assets for the Streamlit / Hugging Face Space app.

Copies small result artifacts and crops the original-input panels from the saved
analysis plots into standalone demo sample images. Checkpoints are *not* copied
unless ``--include-checkpoints`` is set, because they are large and are better
tracked with Git LFS or mounted externally for local Docker.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_LAYOUT = {
    "full": {
        "artifact_dir": "Full",
        "result_subdir": "drive-download-20260502T223147Z-3-001",
        "plots": [
            "analysis_sample_0.png",
            "analysis_sample_1.png",
            "analysis_sample_2.png",
            "confusion_matrices.png",
            "illumination_invariance.png",
            "leaf_mask_sanity.png",
            "training_curves.png",
        ],
        "results": [
            "history.json",
            "phasephyto_domain_shift.json",
            "phasephyto_results.json",
            "target_classification_report.txt",
        ],
        "checkpoint_files": ["best_phasephyto.pt", "final_ema_phasephyto.pt"],
    },
    "backbone_only": {
        "artifact_dir": "Backbone only",
        "result_subdir": "drive-download-20260502T223311Z-3-001",
        "plots": ["training_curves.png"],
        "results": ["history.json", "phasephyto_domain_shift.json", "target_classification_report.txt"],
        "checkpoint_files": ["best_phasephyto.pt"],
    },
    "no_fusion": {
        "artifact_dir": "No_fusion",
        "result_subdir": "drive-download-20260502T223242Z-3-001",
        "plots": ["training_curves.png"],
        "results": ["history.json", "phasephyto_domain_shift.json", "target_classification_report.txt"],
        "checkpoint_files": ["best_phasephyto.pt"],
    },
    "pc_only": {
        "artifact_dir": "PC only",
        "result_subdir": "drive-download-20260502T223439Z-3-001",
        "plots": ["training_curves.png"],
        "results": ["phasephyto_domain_shift.json", "target_classification_report.txt"],
        "checkpoint_files": ["best_phasephyto.pt"],
    },
    "baseline": {
        "artifact_dir": ".",
        "result_subdir": "",
        "plots": [],
        "results": [],
        "checkpoint_files": ["baseline_vit.pt"],
    },
}

CORRECT_SAMPLES = [
    {
        "key": "apple_scab_good_1",
        "source_plot": "Full/drive-download-20260502T223147Z-3-001/plots/analysis_sample_0.png",
        "label": "Apple___Apple_scab",
        "label_index": 0,
        "known_good_models": ["full", "full_ema", "backbone_only", "no_fusion", "baseline"],
    },
    {
        "key": "apple_scab_good_2",
        "source_plot": "Full/drive-download-20260502T223147Z-3-001/plots/analysis_sample_2.png",
        "label": "Apple___Apple_scab",
        "label_index": 0,
        "known_good_models": ["full", "full_ema", "backbone_only", "no_fusion", "baseline"],
    },
]


def crop_input_panel(source_plot: Path, dest_image: Path) -> None:
    image = Image.open(source_plot).convert("RGB")
    arr = np.array(image)
    left_panel = arr[:, : max(1, arr.shape[1] // 5)]
    non_white = np.where((left_panel < 245).any(axis=2))
    if non_white[0].size == 0 or non_white[1].size == 0:
        crop = left_panel
    else:
        y0, y1 = int(non_white[0].min()), int(non_white[0].max())
        x0, x1 = int(non_white[1].min()), int(non_white[1].max())
        crop = left_panel[y0 : y1 + 1, x0 : x1 + 1]
    dest_image.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop).save(dest_image)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to Final Project/Finalresults")
    parser.add_argument("--dest", default="hf_assets/finalresults", help="Destination asset root")
    parser.add_argument("--include-checkpoints", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    for spec in MODEL_LAYOUT.values():
        artifact_root = source / spec["artifact_dir"] if spec["artifact_dir"] != "." else source
        run_root = artifact_root / spec["result_subdir"] if spec["result_subdir"] else None
        for name in spec["plots"]:
            src = run_root / "plots" / name
            if src.exists():
                copy_file(src, dest / spec["artifact_dir"] / spec["result_subdir"] / "plots" / name)
        for name in spec["results"]:
            src = run_root / "results" / name
            if src.exists():
                copy_file(src, dest / spec["artifact_dir"] / spec["result_subdir"] / "results" / name)
        if args.include_checkpoints:
            for name in spec["checkpoint_files"]:
                src = artifact_root / name
                if src.exists():
                    copy_file(src, dest / spec["artifact_dir"] / name)

    samples = []
    samples_dir = Path("hf_assets/samples")
    for sample in CORRECT_SAMPLES:
        source_plot = source / sample["source_plot"]
        dest_image = samples_dir / f"{sample['key']}.png"
        crop_input_panel(source_plot, dest_image)
        samples.append(
            {
                **sample,
                "image_path": str(Path("samples") / f"{sample['key']}.png"),
                "source_plot_path": str(Path("finalresults") / sample["source_plot"]),
            }
        )

    manifest = {"samples": samples}
    Path("hf_assets/demo_samples.json").write_text(json.dumps(manifest, indent=2))
    print(f"Prepared assets under {dest}")


if __name__ == "__main__":
    main()
