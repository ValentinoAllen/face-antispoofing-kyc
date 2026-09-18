"""Build the size- and quality-normalized image cache the cached baseline trains on.

Run on Kaggle from the repository root, with the ``antispoof`` package importable:

    python scripts/build_cache.py --data-config configs/data.yaml --config configs/cache_v1.yaml \
        --manifest-dir /kaggle/working/manifests --splits train,val \
        --output-dir /kaggle/working/cache [--limit N] [--dataset-root PATH]

Writes <output-dir>/<split>/<subject_id>/<live|spoof>/<filename>, one cache_manifest_<split>.csv per
split and cache_summary.json. Nothing under the dataset root is written. The build is idempotent: a
cached file whose source SHA-256 still matches is skipped, so rerunning the same command resumes an
interrupted build. This writes no run record and prints no ledger row.
"""

import argparse
import dataclasses
import logging
from pathlib import Path

from antispoof.data.cache_build import CacheInputs, build_cache, format_report, load_cache_config
from antispoof.data.config import load_data_config
from antispoof.training.reproducibility import find_repo_root


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. Path flags override the matching data config values."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Cache config.")
    parser.add_argument(
        "--manifest-dir", type=Path, required=True, help="Overrides data.manifest_dir."
    )
    parser.add_argument(
        "--splits", required=True, help="Comma-separated splits to cache, e.g. train,val."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="The cache root to write.")
    parser.add_argument("--limit", type=int, help="Cache only the first N rows of each split.")
    parser.add_argument("--dataset-root", type=Path, help="Overrides data.dataset_root.")
    return parser.parse_args()


def main() -> None:
    """Load the configs, apply path overrides, build the cache and print the report."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides = {"manifest_dir": args.manifest_dir, "dataset_root": args.dataset_root}
    data_config = dataclasses.replace(
        load_data_config(args.data_config),
        **{key: value for key, value in overrides.items() if value is not None},
    )
    report = build_cache(
        CacheInputs(
            config=load_cache_config(args.config),
            config_path=args.config,
            data_config=data_config,
            splits=tuple(split.strip() for split in args.splits.split(",") if split.strip()),
            output_dir=args.output_dir,
            repo_root=find_repo_root(Path.cwd()),
            limit=args.limit,
        )
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
