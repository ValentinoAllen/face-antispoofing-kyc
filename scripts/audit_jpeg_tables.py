"""Audit the JPEG quantization tables of every manifest row, from image headers only.

Run on Kaggle from the repository root, with the ``antispoof`` package importable:

    python scripts/audit_jpeg_tables.py --data-config configs/data.yaml \
        --config configs/audit_jpeg.yaml \
        --manifest-dir /kaggle/working/manifests --output-dir /kaggle/working/runs \
        [--dataset-root PATH]

Writes <output-dir>/<run_id>/ (record.json, resolved_config.json and audit_summary.json). No pixel
data is decoded. The test manifest's headers are read, which does not spend the single test
evaluation of docs/RULES.md 3: no model is built and no model-selection signal is produced. This run
computes no PAD metrics, so it gets no docs/EXPERIMENTS.md row and no ledger row is printed.
"""

import argparse
import dataclasses
import logging
from pathlib import Path

from antispoof.data.config import load_data_config
from antispoof.eval.jpeg_audit import (
    AuditInputs,
    format_audit_report,
    load_audit_config,
    run_audit,
)
from antispoof.training.reproducibility import find_repo_root


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. Path flags override the matching data config values."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Audit config.")
    parser.add_argument(
        "--manifest-dir", type=Path, required=True, help="Overrides data.manifest_dir."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Parent of the run dir.")
    parser.add_argument("--dataset-root", type=Path, help="Overrides data.dataset_root.")
    return parser.parse_args()


def main() -> None:
    """Load the configs, apply path overrides, run the audit, then print the report.

    No ledger row is printed: the audit computes no PAD metrics.
    """
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides = {"manifest_dir": args.manifest_dir, "dataset_root": args.dataset_root}
    data_config = dataclasses.replace(
        load_data_config(args.data_config),
        **{key: value for key, value in overrides.items() if value is not None},
    )
    result = run_audit(
        AuditInputs(
            config=load_audit_config(args.config),
            config_path=args.config,
            data_config=data_config,
            output_dir=args.output_dir,
            repo_root=find_repo_root(Path.cwd()),
        )
    )
    print(format_audit_report(result))


if __name__ == "__main__":
    main()
