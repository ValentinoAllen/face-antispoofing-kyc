"""Train the baseline on a train-manifest subset and evaluate it on a val-manifest subset.

Run on Kaggle from the repository root, with the ``antispoof`` package importable:

    python scripts/train.py --data-config configs/data.yaml --config configs/baseline.yaml \
        --manifest-dir /kaggle/working/manifests --output-dir /kaggle/working/runs

Writes <output-dir>/<run_id>/ (record.json, checkpoint.pt, predictions.csv, resolved_config.json).
The last line printed is one docs/EXPERIMENTS.md row. The test split is not read.
"""

import argparse
import dataclasses
import logging
from pathlib import Path

from antispoof.data.config import load_data_config
from antispoof.training.config import load_train_config
from antispoof.training.reproducibility import find_repo_root
from antispoof.training.run import RunInputs, format_ledger_row, format_run_summary, run_baseline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. Path flags override the matching data config values."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Experiment config.")
    parser.add_argument(
        "--manifest-dir", type=Path, required=True, help="Overrides data.manifest_dir."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Parent of the run dir.")
    parser.add_argument("--dataset-root", type=Path, help="Overrides data.dataset_root.")
    return parser.parse_args()


def main() -> None:
    """Load both configs, apply path overrides, run, and print the summary and ledger row."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides = {"manifest_dir": args.manifest_dir, "dataset_root": args.dataset_root}
    data_config = dataclasses.replace(
        load_data_config(args.data_config),
        **{key: value for key, value in overrides.items() if value is not None},
    )
    result = run_baseline(
        RunInputs(
            train_config=load_train_config(args.config),
            config_path=args.config,
            data_config=data_config,
            output_dir=args.output_dir,
            repo_root=find_repo_root(Path.cwd()),
        )
    )
    print(format_run_summary(result))
    print(format_ledger_row(result.record))


if __name__ == "__main__":
    main()
