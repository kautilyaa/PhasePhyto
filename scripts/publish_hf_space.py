#!/usr/bin/env python3
"""Publish the PhasePhyto demo to Hugging Face.

Creates/updates:
- a public model repo containing checkpoints
- a public Docker Space containing the Streamlit app

Usage:
  HF_TOKEN=hf_xxx python scripts/publish_hf_space.py \
      --username Mathesh0803 \
      --space-name phasephyto-explorer \
      --weights-name phasephyto-weights \
      --weights-source '/Users/matheshwara/Downloads/DATA640_CV/Final Project/Finalresults'
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE_INCLUDE = [
    'README.md',
    'Dockerfile',
    'pyproject.toml',
    'streamlit_app.py',
    'phasephyto',
    'hf_assets',
]


def copy_into(src: Path, dst: Path) -> None:
    if src.name == '.DS_Store':
        return
    if src.is_dir():
        shutil.copytree(
            src,
            dst,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns('.DS_Store'),
        )
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_space_bundle(dest: Path) -> None:
    for rel in SPACE_INCLUDE:
        src = ROOT / rel
        if src.exists():
            copy_into(src, dest / rel)


def build_weights_bundle(weights_source: Path, dest: Path) -> None:
    mapping = {
        weights_source / 'Full' / 'best_phasephyto.pt': dest / 'full' / 'best_phasephyto.pt',
        weights_source / 'Full' / 'final_ema_phasephyto.pt': dest / 'full' / 'final_ema_phasephyto.pt',
        weights_source / 'Backbone only' / 'best_phasephyto.pt': dest / 'backbone_only' / 'best_phasephyto.pt',
        weights_source / 'No_fusion' / 'best_phasephyto.pt': dest / 'no_fusion' / 'best_phasephyto.pt',
        weights_source / 'baseline_vit.pt': dest / 'baseline' / 'baseline_vit.pt',
    }
    for src, dst in mapping.items():
        if not src.exists():
            raise FileNotFoundError(f'Missing required checkpoint: {src}')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    readme = dest / 'README.md'
    readme.write_text(
        '# PhasePhyto Weights\n\n'
        'Public checkpoints used by the `Mathesh0803/phasephyto-explorer` Hugging Face Space.\n\n'
        'Files included:\n'
        '- `full/best_phasephyto.pt`\n'
        '- `full/final_ema_phasephyto.pt`\n'
        '- `backbone_only/best_phasephyto.pt`\n'
        '- `no_fusion/best_phasephyto.pt`\n'
        '- `baseline/baseline_vit.pt`\n'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', required=True)
    parser.add_argument('--space-name', default='phasephyto-explorer')
    parser.add_argument('--weights-name', default='phasephyto-weights')
    parser.add_argument('--weights-source', required=True)
    parser.add_argument('--token', default=os.environ.get('HF_TOKEN'))
    args = parser.parse_args()

    if not args.token:
        raise SystemExit('Provide a Hugging Face token via --token or HF_TOKEN')

    api = HfApi(token=args.token)
    space_repo = f'{args.username}/{args.space_name}'
    weights_repo = f'{args.username}/{args.weights_name}'
    weights_source = Path(args.weights_source)

    api.create_repo(repo_id=weights_repo, private=False, repo_type='model', exist_ok=True)
    api.create_repo(repo_id=space_repo, private=False, repo_type='space', space_sdk='docker', exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        weights_bundle = tmp / 'weights'
        space_bundle = tmp / 'space'
        build_weights_bundle(weights_source, weights_bundle)
        build_space_bundle(space_bundle)

        api.upload_folder(
            folder_path=str(weights_bundle),
            repo_id=weights_repo,
            repo_type='model',
            commit_message='Upload PhasePhyto checkpoints',
        )
        api.upload_folder(
            folder_path=str(space_bundle),
            repo_id=space_repo,
            repo_type='space',
            commit_message='Deploy PhasePhyto Space',
        )

    print(f'Space URL: https://huggingface.co/spaces/{space_repo}')
    print(f'Weights URL: https://huggingface.co/{weights_repo}')


if __name__ == '__main__':
    main()
