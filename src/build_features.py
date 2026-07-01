"""Build the modeling-ready feature matrix from the leak-safe base dataset.

Reads ``data/processed/base_dataset.csv`` (produced by ``make_dataset.py``),
applies the family-organized feature layer, tags each row with its temporal
``split``, and writes ``data/processed/modeling_dataset.csv``.

Features are computed over **all** events (history-only events included) so that
prior-event / recent-form / head-to-head signals are populated, but only rows
from the labeled events (train/val/test) are emitted — history-only events serve
purely as feature context. This step is orthogonal to data collection: it never
touches the network and only consumes pre-game columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from events import SPLIT_BY_EVENT, TARGET_ROLES
from features import build_feature_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def main() -> None:
    """Build and write the modeling dataset."""
    args = parse_args()
    processed_dir = args.data_dir / "processed"

    base = pd.read_csv(processed_dir / "base_dataset.csv")
    features = build_feature_matrix(base)

    # Tag rows with their temporal split; keep only labeled (non-history) events.
    features["split"] = features["event"].map(SPLIT_BY_EVENT)
    modeling = features[features["split"].isin(TARGET_ROLES)].reset_index(drop=True)

    output_path = processed_dir / "modeling_dataset.csv"
    modeling.to_csv(output_path, index=False)

    split_counts = modeling["split"].value_counts().to_dict()
    print(
        "Built modeling dataset: "
        f"{len(modeling)} rows, {modeling.shape[1]} columns -> {output_path} "
        f"| splits: {split_counts}"
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory holding the processed datasets.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
