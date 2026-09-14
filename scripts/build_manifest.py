"""Build manifest_{train,val,test}.csv and split_assignment.csv from configs/data.yaml.

Run on Kaggle from the repository root, with the ``antispoof`` package importable:

    python scripts/build_manifest.py --config configs/data.yaml --output-dir data/manifests

The printed counts are compared against the externally measured figures in docs/SCHEMA.md §1.
"""

import argparse
import dataclasses
import logging
from pathlib import Path

from antispoof.data.build import format_report, run
from antispoof.data.config import load_data_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. Path flags override the matching config values."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--dataset-root", type=Path, help="Overrides data.dataset_root.")
    parser.add_argument("--output-dir", type=Path, help="Overrides data.manifest_dir.")
    parser.add_argument(
        "--split-assignment", type=Path, help="Overrides data.split_assignment_path."
    )
    return parser.parse_args()


def main() -> None:
    """Load the config, apply path overrides, build everything and print the report."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides = {
        "dataset_root": args.dataset_root,
        "manifest_dir": args.output_dir,
        "split_assignment_path": args.split_assignment,
    }
    config = dataclasses.replace(
        load_data_config(args.config),
        **{key: value for key, value in overrides.items() if value is not None},
    )
    print(format_report(run(config)))


if __name__ == "__main__":
    main()
