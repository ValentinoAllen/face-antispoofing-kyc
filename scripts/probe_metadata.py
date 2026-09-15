"""Probe whether image-header metadata alone separates live from spoof on the baseline's subsets.

Run on Kaggle from the repository root, with the ``antispoof`` package importable:

    python scripts/probe_metadata.py --data-config configs/data.yaml \
        --config configs/probe_metadata.yaml \
        --manifest-dir /kaggle/working/manifests --output-dir /kaggle/working/runs \
        [--predictions PATH] [--dataset-root PATH]

Writes <output-dir>/<run_id>/ (record.json, resolved_config.json, probe_summary.json and
features.csv). No pixel data is decoded and the test split is not read. The analysis is printed
first; the last line printed is one docs/EXPERIMENTS.md row for the primary classifier.
"""

import argparse
import dataclasses
import logging
from pathlib import Path

from antispoof.data.config import load_data_config
from antispoof.eval.metadata_probe import (
    ProbeInputs,
    format_probe_report,
    load_probe_config,
    run_probe,
)
from antispoof.training.config import load_train_config
from antispoof.training.reproducibility import find_repo_root
from antispoof.training.run import format_ledger_row


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. Path flags override the matching data config values."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Probe config.")
    parser.add_argument(
        "--manifest-dir", type=Path, required=True, help="Overrides data.manifest_dir."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Parent of the run dir.")
    parser.add_argument(
        "--predictions", type=Path, help="A baseline predictions.csv to compare with."
    )
    parser.add_argument("--dataset-root", type=Path, help="Overrides data.dataset_root.")
    return parser.parse_args()


def main() -> None:
    """Load the configs, apply path overrides, run the probe, then print the report and ledger row.

    The ledger row is printed last.
    """
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides = {"manifest_dir": args.manifest_dir, "dataset_root": args.dataset_root}
    data_config = dataclasses.replace(
        load_data_config(args.data_config),
        **{key: value for key, value in overrides.items() if value is not None},
    )
    probe_config = load_probe_config(args.config)
    subset_config_path = Path(probe_config.subsets.config)
    result = run_probe(
        ProbeInputs(
            probe_config=probe_config,
            config_path=args.config,
            subset_config=load_train_config(subset_config_path),
            subset_config_path=subset_config_path,
            data_config=data_config,
            output_dir=args.output_dir,
            repo_root=find_repo_root(Path.cwd()),
            predictions_path=args.predictions,
        )
    )
    print(format_probe_report(result))
    print(format_ledger_row(result.record))


if __name__ == "__main__":
    main()
