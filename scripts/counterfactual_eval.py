"""Evaluate a baseline checkpoint with attack images re-encoded at evaluation time.

Run on Kaggle from the repository root, with the ``antispoof`` package importable:

    python scripts/counterfactual_eval.py --data-config configs/data.yaml \
        --config configs/counterfactual_jpeg.yaml \
        --manifest-dir /kaggle/working/manifests --run-dir /kaggle/working/runs/<baseline_run_id> \
        --output-dir /kaggle/working/runs [--dataset-root PATH]

--run-dir is a baseline run folder (checkpoint.pt, resolved_config.json, record.json) whose
config_hash must match the committed baseline config. Writes <output-dir>/<run_id>/ (record.json,
resolved_config.json, counterfactual_summary.json and predictions_<arm>.csv). The quantization-table
audit and a table of all arms are printed first; the last line printed is one docs/EXPERIMENTS.md
row for the reencode_live_quality arm. The test split is not read.
"""

import argparse
import dataclasses
import logging
from pathlib import Path

from antispoof.data.config import load_data_config
from antispoof.eval.counterfactual import (
    CounterfactualInputs,
    format_counterfactual_report,
    load_counterfactual_config,
    run_counterfactual,
)
from antispoof.training.config import load_train_config
from antispoof.training.reproducibility import find_repo_root
from antispoof.training.run import format_ledger_row


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. Path flags override the matching data config values."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Counterfactual config.")
    parser.add_argument(
        "--manifest-dir", type=Path, required=True, help="Overrides data.manifest_dir."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Baseline run folder.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Parent of the run dir.")
    parser.add_argument("--dataset-root", type=Path, help="Overrides data.dataset_root.")
    return parser.parse_args()


def main() -> None:
    """Load the configs, apply path overrides, run, then print the report and the ledger row.

    The ledger row is printed last.
    """
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides = {"manifest_dir": args.manifest_dir, "dataset_root": args.dataset_root}
    data_config = dataclasses.replace(
        load_data_config(args.data_config),
        **{key: value for key, value in overrides.items() if value is not None},
    )
    config = load_counterfactual_config(args.config)
    baseline_config_path = Path(config.baseline.config)
    result = run_counterfactual(
        CounterfactualInputs(
            config=config,
            config_path=args.config,
            baseline_config=load_train_config(baseline_config_path),
            baseline_config_path=baseline_config_path,
            data_config=data_config,
            source_run_dir=args.run_dir,
            output_dir=args.output_dir,
            repo_root=find_repo_root(Path.cwd()),
        )
    )
    print(format_counterfactual_report(result))
    print(format_ledger_row(result.record))


if __name__ == "__main__":
    main()
