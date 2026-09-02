#!/usr/bin/env python3
"""Fail CI when a candidate model's MAE regresses beyond the allowed threshold."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any


def load_mae(path: Path) -> float:
    """Load an MAE value from a JSON metrics file or persisted model bundle."""
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file does not exist: {path}")

    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as file:
            payload: Any = json.load(file)
    else:
        with path.open("rb") as file:
            payload = pickle.load(file)

    if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
        payload = payload["metrics"]

    try:
        mae = float(payload["mae"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not find a numeric 'mae' in {path}") from exc

    if mae < 0:
        raise ValueError(f"MAE must be non-negative in {path}: {mae}")
    return mae


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail if the new model's MAE is more than 5%% worse than production."
    )
    parser.add_argument("--new-metrics", type=Path, required=True)
    parser.add_argument("--production-metrics", type=Path, required=True)
    parser.add_argument(
        "--max-regression-percent",
        type=float,
        default=5.0,
        help="Maximum allowed MAE regression percentage (default: 5).",
    )
    args = parser.parse_args()

    try:
        new_mae = load_mae(args.new_metrics)
        production_mae = load_mae(args.production_metrics)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, pickle.UnpicklingError) as exc:
        print(f"MAE check could not run: {exc}", file=sys.stderr)
        return 1

    if production_mae == 0:
        percent_change = 0.0 if new_mae == 0 else float("inf")
    else:
        percent_change = ((new_mae - production_mae) / production_mae) * 100

    print(
        f"Production MAE: {production_mae:.6f}; "
        f"new MAE: {new_mae:.6f}; "
        f"percent change: {percent_change:.2f}%"
    )

    if percent_change > args.max_regression_percent:
        print(
            f"MAE regression exceeds {args.max_regression_percent:.2f}%; failing CI.",
            file=sys.stderr,
        )
        return 1

    print("MAE regression is within the allowed threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
